#!/usr/bin/env bash
# Everything that is trained or scored for the "big cross-encoder + pass-2" upgrade. Run on the AWS GPU instance (g5.2xlarge),
# from the repository root, after aws/setup.sh and after normalized/ + models/ of the laptop run are in $ER_ROOT.
#
#   bash aws/run_pipeline.sh pilot      ~0.5 h   big cross-encoder pilot on 1,500 steps + go/no-go numbers (cheap, run this first)
#   bash aws/run_pipeline.sh bigce      ~4 h     full big cross-encoder -> ranker B4 (LightGBM with both cross-encoders) + held-out comparison
#   bash aws/run_pipeline.sh mirror     ~4-7 h   test-time pipeline over the TRAINING records (full-density table for pass 2)
#   bash aws/run_pipeline.sh pass2      ~1 h     fit pass 2 on the mirror, held-out comparison with pass 1
#   bash aws/run_pipeline.sh finish     ~3-4 h   real test set: big cross-encoder scores, pass-1 scores, pass 2, submission files, validator
#
# P1=b2 (default) uses the deployed LightGBM + small cross-encoder as pass 1; P1=b4 uses the model from `bigce`.
# Every step is resumable (skipped when its output exists) and logs to $ER_ROOT/logs/<step>.log (mirror: $ER_ROOT/fulltrain/logs).
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONUTF8=1 ER_ROOT="${ER_ROOT:-/data/er_work}" ER_DATASET="${ER_DATASET:-/data/dataset}"
export ER_CHUNK=50000 ER_THREADS="$(nproc)" ER_WORKERS="$(( $(nproc) > 2 ? $(nproc) - 1 : 2 ))"
export ER_SEARCH_BATCH=1024 ER_ENCODE_BATCH=2048          # 24 GB card
export ER_CE_BATCH=32 ER_CE_ACCUM=2 ER_CE_LR=2e-5 ER_CE_DTYPE=bf16 ER_CE_SCORE_BATCH=512
export ER_CE_BASE="${ER_CE_BASE:-BAAI/bge-reranker-v2-m3}"  # Apache-2.0, 568M parameters
P1="${P1:-b2}"
REAL="$ER_ROOT"; MIR="$ER_ROOT/fulltrain"
PHASE="${1:-}"

# run <root> <name> <done-file> <cmd...>   (env assignments go through `env` in the command)
run() {
  local root=$1 name=$2 done=$3; shift 3
  mkdir -p "$root/logs"
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] already done"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"
  ER_ROOT="$root" "$@" >> "$root/logs/$name.log" 2>&1 || { echo "$(date +%H:%M:%S) [$name] FAILED, see $root/logs/$name.log"; exit 1; }
  echo "$(date +%H:%M:%S) [$name] done"
}
SMALL=(ER_CE_TAG=_v2 ER_CE_MODEL_DIR=ce_er2)
BIGCE=(ER_CE_TAG=_big ER_CE_MODEL_DIR=ce_big)
BAND=(ER_STAGE1=ranker_a.txt ER_FEAT_TAG=_s)             # the uncertain band is defined by ranker A, as before
B4=(ER_CE=1 ER_CE_TAG=_v2 ER_CE2_TAG=_big ER_MODEL=ranker_b4.txt)
B2=(ER_CE=1 ER_CE_TAG=_v2 ER_MODEL=ranker_b2.txt)
if [ "$P1" = b4 ]; then FINAL=("${B4[@]}"); PRED=pred_final4; else FINAL=("${B2[@]}"); PRED=pred_final; fi

mine() {   # hard pairs w.r.t. B2 (its twins and borderline cases), all training records
  run "$REAL" mine_big "$REAL/normalized/cepairs_big.parquet" env ER_CE_TAG=_big ER_STAGE1=ranker_b2.txt ER_FEAT_TAG=_ce2 ER_CE_SAMPLE=1.0 python -u crossenc.py mine
}

case "$PHASE" in
pilot)
  mine
  run "$REAL" train_pilot "$REAL/models/ce_pilot/model.safetensors" env ER_CE_TAG=_big ER_CE_MODEL_DIR=ce_pilot ER_CE_STEPS=1500 ER_CE_EXTRA_PAIRS=_v2 python -u crossenc.py train
  run "$REAL" score_eval_pilot "$REAL/normalized/ce_eval_pilot.parquet" env ER_CE_TAG=_pilot ER_CE_MODEL_DIR=ce_pilot "${BAND[@]}" python -u crossenc.py score eval
  ER_ROOT="$REAL" python aws/compare_ce.py _v2 _pilot | tee "$REAL/logs/pilot_compare.txt"
  ;;
bigce)
  mine
  run "$REAL" train_big "$REAL/models/ce_big/model.safetensors" env "${BIGCE[@]}" ER_CE_EPOCHS=2 ER_CE_EXTRA_PAIRS=_v2 python -u crossenc.py train
  for sp in train eval; do
    run "$REAL" score_${sp}_big "$REAL/normalized/ce_${sp}_big.parquet" env "${BIGCE[@]}" "${BAND[@]}" python -u crossenc.py score $sp
  done
  for sp in train eval; do
    run "$REAL" features_${sp}_b4 "$REAL/normalized/feat_${sp}_ce4.parquet" env "${B4[@]}" ER_FEAT_TAG=_ce4 python -u ranker.py features $sp
  done
  run "$REAL" fit_b4 "$REAL/models/decision_b4.json" env "${B4[@]}" ER_FEAT_TAG=_ce4 ER_DECISION=decision_b4.json python -u ranker.py fit
  ER_ROOT="$REAL" python - <<'PY' | tee "$REAL/logs/b4_compare.txt"
import json, os
m = os.environ["ER_ROOT"] + "/models/"
for n in ("decision_b2.json", "decision_b4.json"):
    h = json.load(open(m + n))["held"]
    print(n, {k: round(v["official"], 4) for k, v in h.items()})
print("USE B4 only if its 'early-stop half' beats B2's by more than 0.0005 (the half no threshold was tuned on).")
PY
  ;;
mirror)
  ER_ROOT="$REAL" python fulltrain_setup.py
  L="$MIR/normalized"
  run "$MIR" block_test        "$L/cand/test_sparse/_DONE"          python -u block.py test
  run "$MIR" embed_search_test "$L/cand/test_dense/_DONE"           python -u embed.py search test
  run "$MIR" extras_sparse     "$L/cand/test_extra_sparse.parquet"  python -u extras.py sparse test
  run "$MIR" extras_dense      "$L/cand/test_extra_dense.parquet"   python -u extras.py dense test
  run "$MIR" extras_build      "$L/cand/test_extra.parquet"         python -u extras.py build test
  run "$MIR" score_a           "$L/pred_a/_DONE"                    env ER_FEAT_TAG=_s ER_MODEL=ranker_a.txt ER_PRED="$L/pred_a" python -u predict.py score
  run "$MIR" ce_test_small     "$L/ce_test_v2.parquet"              env "${SMALL[@]}" "${BAND[@]}" ER_PRED="$L/pred_a" python -u crossenc.py score test
  if [ "$P1" = b4 ]; then
    run "$MIR" ce_test_big     "$L/ce_test_big.parquet"             env "${BIGCE[@]}" "${BAND[@]}" ER_PRED="$L/pred_a" python -u crossenc.py score test
  fi
  run "$MIR" score_final       "$L/$PRED/_DONE"                     env "${FINAL[@]}" ER_PRED="$L/$PRED" python -u predict.py score
  ;;
pass2)
  run "$MIR" pass2_fit "$MIR/models/pass2.txt" env ER_PASS1="$MIR/normalized/$PRED" ER_INSAMPLE_DIR="$REAL/normalized" python -u pass2.py fit
  cp -f "$MIR/models/pass2.txt" "$MIR/models/decision_pass2.json" "$REAL/models/"
  grep -E "RESULT|pass 2 minus" "$MIR/logs/pass2_fit.log" | tee "$REAL/logs/pass2_compare.txt"
  ;;
finish)
  N="$REAL/normalized"
  if [ "$P1" = b4 ]; then
    run "$REAL" ce_test_big "$N/ce_test_big.parquet" env "${BIGCE[@]}" "${BAND[@]}" ER_PRED="$N/pred_a" python -u crossenc.py score test
  fi
  run "$REAL" score_final "$N/$PRED/_DONE" env "${FINAL[@]}" ER_PRED="$N/$PRED" python -u predict.py score
  run "$REAL" pass2_apply "$N/pred_pass2/_DONE" env ER_PASS1="$N/$PRED" python -u pass2.py apply
  run "$REAL" write_pass2 "$REAL/output_pass2/matching_results.tsv" env ER_DECISION=decision_pass2.json ER_PRED="$N/pred_pass2" ER_OUT="$REAL/output_pass2" python -u predict.py write
  python "$(dirname "$ER_DATASET")/utils/validate_submission.py" --matching "$REAL/output_pass2/matching_results.tsv" \
         --candidate "$REAL/output_pass2/candidate_pairs.tsv" --test-dir "$ER_DATASET/test" | tee "$REAL/output_pass2/validator.log"
  ;;
*)
  sed -n 2,13p "$0"; exit 2 ;;
esac
echo "$(date +%H:%M:%S) phase $PHASE finished"
