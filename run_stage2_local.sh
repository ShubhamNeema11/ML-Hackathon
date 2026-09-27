#!/usr/bin/env bash
# STAGE 2, laptop part: after aws/stage2_gpu.sh, unpack stage2_gpu_results.tgz into this folder (tar xzf stage2_gpu_results.tgz), then
#   bash run_stage2_local.sh          (resumable; logs in logs/stage2_*.log; nothing existing is overwritten: all new files are tagged _m3)
# 1. features + ranker "M3" (B2's recipe; new dense channel + the new reranker score as a second cross-encoder feature) -> held-out vs B2
# 2. no-address specialist on the new candidates (tuned settings) and the joint decision -> held-out vs the current 0.9882      = THE GATE
# 3. only if the gate passes (> +0.0005): test scoring, specialist patch, output_m3/ (+ validator); with pass 2 as output_m3_p2/ (pass 2 was
#    fitted on the current pipeline's scores: its effect on the new scores is not measured, so output_m3/ is the safe one)
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_CHUNK=50000 ER_DENSE_TAG=_m3 ER_NA_TFIDF=1
N=normalized; mkdir -p logs
M3=(env ER_CE=1 ER_CE_TAG=_v2 ER_CE2_TAG=_rr ER_FEAT_TAG=_m3)
NA=(env ER_NOADDR_MODEL=noaddr_m3.txt ER_B2=ranker_m3.txt ER_FEAT_TAG=_m3)
step() { local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "logs/stage2_$name.log" 2>&1; then echo "[$name] FAILED:"; tail -n 15 "logs/stage2_$name.log"; exit 1; fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"; }
for f in "$N/eval_dense_m3.parquet" "$N/cand/train_dense_m3/_DONE" "$N/cand/trainall_dense_m3/_DONE" "$N/cand/test_dense_m3/_DONE" "$N/ce_train_rr.parquet" "$N/ce_eval_rr.parquet" "$N/ce_test_rr.parquet"; do
  [ -e "$f" ] || { echo "missing $f: unpack stage2_gpu_results.tgz here first"; exit 1; }; done
step feat_train  "$N/feat_train_m3.parquet"            "${M3[@]}" python -u ranker.py features train
step feat_eval   "$N/feat_eval_m3.parquet"             "${M3[@]}" python -u ranker.py features eval
step fit_m3      "models/decision_m3.json"             "${M3[@]}" ER_MODEL=ranker_m3.txt ER_DECISION=decision_m3.json python -u ranker.py fit
python -c "import json; b=json.load(open('models/decision_b2.json'))['held']; m=json.load(open('models/decision_m3.json'))['held']; print('ranker alone, held-out report half: B2', round(b['early-stop half']['official'],4), '-> M3', round(m['early-stop half']['official'],4), '(tuned half', round(b['thr-half']['official'],4), '->', round(m['thr-half']['official'],4), ')')" | tee logs/stage2_gate.txt
step na_trainall "$N/feat_noaddr2_m3_trainall.parquet" python -u noaddr.py features trainall
step na_eval     "$N/feat_noaddr2_m3_eval.parquet"     python -u noaddr.py features eval
step na_fit      "models/noaddr_m3.txt"                "${NA[@]}" ER_NA_LR=0.05 ER_NA_LEAVES=127 ER_NA_MINLEAF=400 ER_NA_ROUNDS=9000 python -u noaddr.py fit
step na_joint    "models/decision_na_m3.json"          "${NA[@]}" ER_DECISION=decision_na_m3.json python -u noaddr.py joint
GATE=$(python -c "
import json
cur = json.load(open('models/decision_na.json'))['held']['early-stop half']['official']
new = json.load(open('models/decision_na_m3.json'))['held']['early-stop half']['official']
print(f'whole held-out, report half: current {cur:.4f} -> new {new:.4f} ({new - cur:+.4f})')
print('PASS' if new > cur + 0.0005 else 'FAIL')")
echo "$GATE" | tee -a logs/stage2_gate.txt
echo "$GATE" | tail -1 | grep -q PASS || { echo "gate FAILED: the new embedder / reranker do not beat the current pipeline; nothing else is run"; exit 0; }
step score_test  "$N/pred_m3/_DONE"                    "${M3[@]}" ER_MODEL=ranker_m3.txt ER_PRED="$N/pred_m3" python -u predict.py score
step na_test     "$N/feat_noaddr2_m3_test.parquet"     python -u noaddr.py features test
step na_apply    "$N/pred_na_m3/_DONE"                 "${NA[@]}" ER_PRED_IN="$N/pred_m3" ER_PRED_OUT="$N/pred_na_m3" python -u noaddr.py apply
step write       "output_m3/matching_results.tsv"      env ER_DECISION=decision_na_m3.json ER_PRED="$N/pred_na_m3" ER_OUT=output_m3 python -u predict.py write
step p2_apply    "$N/pred_p2_m3/_DONE"                 env ER_PASS1="$N/pred_na_m3" ER_PASS2_OUT="$N/pred_p2_m3" python -u pass2.py apply
step p2_write    "output_m3_p2/matching_results.tsv"   env ER_DECISION=decision_pass2.json ER_PRED="$N/pred_p2_m3" ER_OUT=output_m3_p2 python -u predict.py write
for o in output_m3 output_m3_p2; do
  python utils/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > $o/validator.log 2>&1
  echo "$o: $(tail -1 $o/validator.log)"; cp logs/stage2_gate.txt $o/GATE.txt
done
echo "$(date +%H:%M:%S) STAGE 2 LOCAL DONE"
