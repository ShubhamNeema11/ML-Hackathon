#!/usr/bin/env bash
# The "big cross-encoder + pass-2" upgrade, unattended, documented, and as cheap as it can be. Run on the AWS GPU instance
# (g5.2xlarge) from the repository root, after `bash aws/setup.sh` and after normalized/ + models/ of the laptop run are in $ER_ROOT.
#
#   tmux new -s er
#   AUTO_STOP=1 S3_OUT=s3://your-bucket/er_results bash aws/run_pipeline.sh all
#
# `all` runs everything with automatic go/no-go gates (each logged in logs/choices.txt):
#   pilot     big cross-encoder for 1,500 steps, compared with the small one          -> GO / NO-GO for the big model
#   bigce     full big cross-encoder, LightGBM B4 with both cross-encoders             -> B4 is used only if it beats B2 by > 0.0005
#   mirror    the test-time pipeline over the TRAINING records (full-density table for pass 2)
#   pass2     fit pass 2 on the mirror, compare with pass 1 (same procedure)          -> used only if it beats pass 1 by > 0.0005
#   finish    real test set: scores, pass 2 (if it won), final files, organisers' validator
# and always ends with deliverables/ (final matchings, RUN_REPORT.md with timings + numbers + sanity checks, logs.tgz).
#
# Single phases:  bash aws/run_pipeline.sh pilot|bigce|mirror|pass2|finish|report      (P1=b2|b4 picks the pass-1 model)
# Options (environment):
#   LEAN=1            skip the big cross-encoder (pass 2 on the existing B2 only): about half the cost
#   AUTO_STOP=1       report + upload + shut the instance down when done (2 min after success, 30 min after a failure; `sudo shutdown -c` cancels)
#   S3_OUT=s3://...   copy deliverables/ there at the end (needs AWS credentials on the instance)
#   MIRROR_OVERLAP=1  (default) run the CPU-only blocking of the mirror while the GPU trains the big cross-encoder
#   FORCE_BIG=1       continue after a pilot NO-GO;   DRYRUN=1  print every command, run nothing
#   PRICE_PER_HOUR    only for the cost estimate in the report (default 1.6)
# Every step is resumable (skipped when its output exists): after an interruption just run the same command again.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONUTF8=1 ER_ROOT="${ER_ROOT:-/data/er_work}" ER_DATASET="${ER_DATASET:-/data/dataset}"
export ER_CHUNK=50000 ER_THREADS="$(nproc)" ER_WORKERS="$(( $(nproc) > 2 ? $(nproc) - 1 : 2 ))"
export ER_SEARCH_BATCH=1024 ER_ENCODE_BATCH=2048          # 24 GB card
export ER_CE_BATCH=32 ER_CE_ACCUM=2 ER_CE_LR=2e-5 ER_CE_DTYPE=bf16 ER_CE_SCORE_BATCH=512
export ER_CE_BASE="${ER_CE_BASE:-BAAI/bge-reranker-v2-m3}"  # Apache-2.0, 568M parameters
REAL="$ER_ROOT"; MIR="$ER_ROOT/fulltrain"
DRYRUN="${DRYRUN:-0}"; LEAN="${LEAN:-0}"; MIRROR_OVERLAP="${MIRROR_OVERLAP:-1}"; AUTO_STOP="${AUTO_STOP:-0}"
PHASE="${1:-}"

# ---------------------------------------------------------------------------------------------- helpers
# run <root> <name> <done-file> <cmd...>: skip when done-file exists; log to <root>/logs/<name>.log; append to the timeline
run() {
  local root=$1 name=$2 done=$3; shift 3
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] already done"; return 0; fi
  if [ "$DRYRUN" = 1 ]; then echo "[dry] $name  (ER_ROOT=$root)  $*"; return 0; fi
  mkdir -p "$root/logs" "$REAL/logs"
  local t0 st; t0=$(date +%s)
  echo "$(date +%H:%M:%S) [$name] start"
  if ER_ROOT="$root" "$@" >> "$root/logs/$name.log" 2>&1; then st=ok; else st=FAILED; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$root" "$name" "$(( $(date +%s) - t0 ))" "$st" >> "$REAL/logs/timeline.tsv"
  if [ "$st" = FAILED ]; then echo "$(date +%H:%M:%S) [$name] FAILED, last lines of $root/logs/$name.log:"; tail -n 8 "$root/logs/$name.log"; exit 1; fi
  echo "$(date +%H:%M:%S) [$name] done ($(( $(date +%s) - t0 )) s)"
}
x() { if [ "$DRYRUN" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }   # any other command
note() { echo "$*"; [ "$DRYRUN" = 1 ] || { mkdir -p "$REAL/logs"; echo "$*" >> "$REAL/logs/choices.txt"; }; }
gate() {
  if [ "$DRYRUN" = 1 ]; then case $1 in pilot) echo go;; b4) echo b4;; pass2) echo pass2;; esac
  else ER_ROOT="$REAL" python aws/gates.py "$1"; fi
}

SMALL=(ER_CE_TAG=_v2 ER_CE_MODEL_DIR=ce_er2)
BIGCE=(ER_CE_TAG=_big ER_CE_MODEL_DIR=ce_big)
BAND=(ER_STAGE1=ranker_a.txt ER_FEAT_TAG=_s)             # the uncertain band is defined by ranker A, as before
B4=(ER_CE=1 ER_CE_TAG=_v2 ER_CE2_TAG=_big ER_MODEL=ranker_b4.txt)
B2=(ER_CE=1 ER_CE_TAG=_v2 ER_MODEL=ranker_b2.txt)
P1="${P1:-b2}"
set_p1() { P1=$1; if [ "$P1" = b4 ]; then FINAL=("${B4[@]}"); PRED=pred_final4; else FINAL=("${B2[@]}"); PRED=pred_final; fi; }
set_p1 "$P1"

# ---------------------------------------------------------------------------------------------- phases
mine() {   # hard pairs w.r.t. B2 (its twins and borderline cases) over all training records
  run "$REAL" mine_big "$REAL/normalized/cepairs_big.parquet" env ER_CE_TAG=_big ER_STAGE1=ranker_b2.txt ER_FEAT_TAG=_ce2 ER_CE_SAMPLE=1.0 python -u crossenc.py mine
}

p_pilot() {
  mine
  run "$REAL" train_pilot "$REAL/models/ce_pilot/model.safetensors" env ER_CE_TAG=_big ER_CE_MODEL_DIR=ce_pilot ER_CE_STEPS=1500 ER_CE_EXTRA_PAIRS=_v2 python -u crossenc.py train
  run "$REAL" score_eval_pilot "$REAL/normalized/ce_eval_pilot.parquet" env ER_CE_TAG=_pilot ER_CE_MODEL_DIR=ce_pilot "${BAND[@]}" python -u crossenc.py score eval
  if [ "$DRYRUN" = 1 ]; then echo "[dry] python aws/compare_ce.py _v2 _pilot"; else
    ER_ROOT="$REAL" python aws/compare_ce.py _v2 _pilot | tee "$REAL/logs/pilot_compare.txt"; fi
}

p_bigce() {
  mine
  run "$REAL" train_big "$REAL/models/ce_big/model.safetensors" env "${BIGCE[@]}" ER_CE_EPOCHS=2 ER_CE_EXTRA_PAIRS=_v2 python -u crossenc.py train
  for sp in train eval; do
    run "$REAL" score_${sp}_big "$REAL/normalized/ce_${sp}_big.parquet" env "${BIGCE[@]}" "${BAND[@]}" python -u crossenc.py score $sp
  done
  for sp in train eval; do
    run "$REAL" features_${sp}_b4 "$REAL/normalized/feat_${sp}_ce4.parquet" env "${B4[@]}" ER_FEAT_TAG=_ce4 python -u ranker.py features $sp
  done
  run "$REAL" fit_b4 "$REAL/models/decision_b4.json" env "${B4[@]}" ER_FEAT_TAG=_ce4 ER_DECISION=decision_b4.json python -u ranker.py fit
  if [ "$DRYRUN" != 1 ]; then ER_ROOT="$REAL" python - <<'PY' | tee "$REAL/logs/b4_compare.txt"
import json, os
m = os.environ["ER_ROOT"] + "/models/"
for n in ("decision_b2.json", "decision_b4.json"):
    h = json.load(open(m + n))["held"]
    print(n, {k: round(v["official"], 4) for k, v in h.items()})
print("B4 is used only if its 'early-stop half' beats B2's by more than 0.0005 (the half no threshold was tuned on).")
PY
  fi
}

p_mirror_cpu() {   # CPU-only part of the mirror: safe to run while the GPU trains
  x env ER_ROOT="$REAL" python fulltrain_setup.py
  local L="$MIR/normalized"
  run "$MIR" block_test        "$L/cand/test_sparse/_DONE"          python -u block.py test
  run "$MIR" extras_sparse     "$L/cand/test_extra_sparse.parquet"  python -u extras.py sparse test
}

p_mirror() {
  x env ER_ROOT="$REAL" python fulltrain_setup.py
  local L="$MIR/normalized"
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
}

p_pass2() {
  run "$MIR" pass2_fit "$MIR/models/pass2.txt" env ER_PASS1="$MIR/normalized/$PRED" ER_INSAMPLE_DIR="$REAL/normalized" python -u pass2.py fit
  if [ "$DRYRUN" = 1 ]; then return 0; fi
  cp -f "$MIR/models/pass2.txt" "$MIR/models/decision_pass2.json" "$REAL/models/"
  grep -E "RESULT|pass 2 minus" "$MIR/logs/pass2_fit.log" | tee "$REAL/logs/pass2_compare.txt"
}

FINAL_DIR=""
p_finish() {
  local N="$REAL/normalized"
  # stage-1 scores of the real test set (upload_to_s3.sh does not upload pred_*): they define which pairs the cross-encoder scores
  run "$REAL" score_a "$N/pred_a/_DONE" env ER_FEAT_TAG=_s ER_MODEL=ranker_a.txt ER_PRED="$N/pred_a" python -u predict.py score
  run "$REAL" ce_test_small "$N/ce_test_v2.parquet" env "${SMALL[@]}" "${BAND[@]}" ER_PRED="$N/pred_a" python -u crossenc.py score test
  if [ "$P1" = b4 ]; then
    run "$REAL" ce_test_big "$N/ce_test_big.parquet" env "${BIGCE[@]}" "${BAND[@]}" ER_PRED="$N/pred_a" python -u crossenc.py score test
  fi
  run "$REAL" score_final "$N/$PRED/_DONE" env "${FINAL[@]}" ER_PRED="$N/$PRED" python -u predict.py score
  local use; use=$(gate pass2)
  if [ "$use" = pass2 ]; then
    note "pass 2: USED (beat pass 1 by more than 0.0005 on the held-out report half)"
    run "$REAL" pass2_apply "$N/pred_pass2/_DONE" env ER_PASS1="$N/$PRED" python -u pass2.py apply
    run "$REAL" write_final "$REAL/output_pass2/matching_results.tsv" env ER_DECISION=decision_pass2.json ER_PRED="$N/pred_pass2" ER_OUT="$REAL/output_pass2" python -u predict.py write
    FINAL_DIR="$REAL/output_pass2"
  else
    note "pass 2: NOT used (no gain above 0.0005 on the held-out report half); writing the pass-1 ($P1) submission"
    run "$REAL" write_final "$REAL/output_pass1/matching_results.tsv" env ER_DECISION=decision_$P1.json ER_PRED="$N/$PRED" ER_OUT="$REAL/output_pass1" python -u predict.py write
    FINAL_DIR="$REAL/output_pass1"
  fi
  local VAL="$(dirname "$ER_DATASET")/utils/validate_submission.py"
  [ -f "$VAL" ] || VAL=$(find "$HOME" "$REAL" -maxdepth 7 -name validate_submission.py 2>/dev/null | head -1)
  if [ -n "$VAL" ] && [ -f "$VAL" ]; then
    x bash -c "python '$VAL' --matching '$FINAL_DIR/matching_results.tsv' --candidate '$FINAL_DIR/candidate_pairs.tsv' --test-dir '$ER_DATASET/test' | tee '$FINAL_DIR/validator.log'"
  else
    note "validator not found: copy validate_submission.py next to the dataset (../utils/) and run it by hand"
  fi
}

p_report() {   # $1 = status text
  [ "$DRYRUN" = 1 ] && { echo "[dry] python aws/report.py (status: $1)"; return 0; }
  [ -z "$FINAL_DIR" ] && for d in "$REAL/output_pass2" "$REAL/output_pass1"; do [ -e "$d/matching_results.tsv" ] && { FINAL_DIR=$d; break; }; done
  ER_ROOT="$REAL" FINAL_DIR="${FINAL_DIR:-$REAL/output_pass1}" STATUS="$1" python aws/report.py || echo "report failed"
  [ -n "${S3_OUT:-}" ] && aws s3 sync "$REAL/deliverables" "$S3_OUT" --only-show-errors && echo "deliverables copied to $S3_OUT"
}

# ---------------------------------------------------------------------------------------------- orchestration
BGPID=""
on_exit() {
  local rc=$?
  [ -n "$BGPID" ] && kill "$BGPID" 2>/dev/null
  [ "$DRYRUN" = 1 ] && return
  trap - EXIT
  if [ $rc -eq 0 ]; then p_report "SUCCESS"; else p_report "FAILED (exit code $rc): see logs/timeline.tsv and the last step log"; fi
  if [ "$AUTO_STOP" = 1 ]; then
    local m=2; [ $rc -ne 0 ] && m=30
    echo "instance stops in $m minutes (cancel with: sudo shutdown -c)"; sudo shutdown -h +$m
  fi
}

case "$PHASE" in
all)
  trap on_exit EXIT
  if [ "$LEAN" = 1 ]; then
    note "LEAN run: the big cross-encoder is skipped; pass 1 = B2"; set_p1 b2
  else
    p_pilot
    g=$(gate pilot); note "pilot gate: $g"
    if [ "$g" = go ] || [ "${FORCE_BIG:-0}" = 1 ]; then
      if [ "$MIRROR_OVERLAP" = 1 ]; then
        ( ER_WORKERS=$(( ER_WORKERS > 4 ? ER_WORKERS - 3 : 2 )); export ER_WORKERS; p_mirror_cpu ) &
        BGPID=$!
      fi
      p_bigce
      if [ -n "$BGPID" ]; then wait "$BGPID" || { echo "background mirror blocking failed"; exit 1; }; BGPID=""; fi
      b=$(gate b4); note "pass-1 model: $b"; set_p1 "$b"
    else
      note "pilot NO-GO: big cross-encoder skipped; pass 1 = B2"; set_p1 b2
    fi
  fi
  p_mirror; p_pass2; p_finish
  ;;
pilot)  p_pilot ;;
bigce)  p_bigce ;;
mirror) p_mirror ;;
pass2)  p_pass2 ;;
finish) p_finish; p_report "finish phase only" ;;
report) p_report "manual" ;;
*)
  sed -n 2,26p "$0"; exit 2 ;;
esac
echo "$(date +%H:%M:%S) phase $PHASE finished"
