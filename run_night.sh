#!/usr/bin/env bash
# Overnight, laptop: pass 2 (sibling support) on top of the quick-path result, with NO retraining of any existing model.
#
#   bash run_night.sh           (repository root; resumable: finished steps are skipped)
#
# 1. mirror folder fulltrain/: the TRAINING records play the test set (hard links, no copies) and go through exactly the pipeline that produced
#    the test scores: sparse + dense blocking, name-only extras, ranker A, cross-encoder band (ce_er2), ranker B2, no-address specialist patch
# 2. pass 2 is fitted there (rows of records any model saw are excluded) and compared with pass 1 on held-out records (thresholds tuned on one
#    half, reported on the other); it is used only if it beats pass 1 by more than 0.0005 on the report half
# 3. if it wins: pass 2 re-scores the real test set (pred_na -> pred_pass2) and output_pass2/ gets the files (+ validator); otherwise nothing changes
# Logs: logs/night_<step>.log, the run log in run_night.out; a summary in output_pass2/RESULT.txt (or NIGHT_RESULT.txt when pass 2 is not used).
set -uo pipefail
cd "$(dirname "$0")"
REPO="$PWD"
export PYTHONUTF8=1 ER_CHUNK=50000
R="$REPO"; MIR="$REPO/fulltrain"; MN="$MIR/normalized"; N="$R/normalized"
mkdir -p "$R/logs"
T0=$(date +%s)
step() {   # step <name> <done-file> <command...>   (commands run with the environment given in front of them)
  local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "$R/logs/night_$name.log" 2>&1; then
    echo "[$name] FAILED, last lines of logs/night_$name.log:"; tail -n 15 "$R/logs/night_$name.log"; exit 1
  fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"
}
for f in "$N/pred_na/_DONE" "$R/models/noaddr_c.txt" "$R/models/ranker_a.txt" "$R/models/ranker_b2.txt" "$R/models/ce_er2/model.safetensors" "$N/trainall_queries.parquet"; do
  [ -e "$f" ] || { echo "missing $f (run run_quick.sh first)"; exit 1; }
done

M_ENV=(env ER_ROOT="$MIR")
# --- mirror (the same commands and settings as the test run that produced pred_final)
step mirror_setup    "$MN/test_source1.parquet"          env ER_ROOT="$R" python -u fulltrain_setup.py
step mirror_sparse   "$MN/cand/test_sparse/_DONE"        "${M_ENV[@]}" python -u block.py test
step mirror_dense    "$MN/cand/test_dense/_DONE"         "${M_ENV[@]}" python -u embed.py search test
step mirror_x_sparse "$MN/cand/test_extra_sparse.parquet" "${M_ENV[@]}" python -u extras.py sparse test
step mirror_x_dense  "$MN/cand/test_extra_dense.parquet" "${M_ENV[@]}" python -u extras.py dense test
step mirror_x_build  "$MN/cand/test_extra.parquet"       "${M_ENV[@]}" python -u extras.py build test
step mirror_score_a  "$MN/pred_a/_DONE"                  "${M_ENV[@]}" ER_FEAT_TAG=_s ER_MODEL=ranker_a.txt ER_PRED="$MN/pred_a" python -u predict.py score
step mirror_ce       "$MN/ce_test_v2.parquet"            "${M_ENV[@]}" ER_CE_TAG=_v2 ER_CE_MODEL_DIR=ce_er2 ER_STAGE1=ranker_a.txt ER_FEAT_TAG=_s ER_PRED="$MN/pred_a" python -u crossenc.py score test
step mirror_score_b2 "$MN/pred_final/_DONE"              "${M_ENV[@]}" ER_CE=1 ER_CE_TAG=_v2 ER_MODEL=ranker_b2.txt ER_PRED="$MN/pred_final" python -u predict.py score
step mirror_na_block "$MN/cand/na_tfidf_test.parquet"    "${M_ENV[@]}" python -u blocking_noaddr.py build test
step mirror_na_feat  "$MN/feat_noaddr2_test.parquet"     "${M_ENV[@]}" ER_NA_TFIDF=1 ER_EXTRAS=0 python -u noaddr.py features test
step mirror_na_apply "$MN/pred_na/_DONE"                 "${M_ENV[@]}" ER_NA_TFIDF=1 ER_PRED_IN="$MN/pred_final" ER_PRED_OUT="$MN/pred_na" python -u noaddr.py apply
# --- pass 2
step pass2_fit       "$MIR/models/pass2.txt"             "${M_ENV[@]}" ER_PASS1="$MN/pred_na" ER_INSAMPLE_DIR="$N" python -u pass2.py fit
echo; grep -h "RESULT\|pass 2 minus" "$R/logs/night_pass2_fit.log" | tail -3
USE=$(python -c "
import json; d = json.load(open('fulltrain/models/decision_pass2.json'))
p2, p1 = d['held']['report-half']['official'], d['pass1_held']['report-half']['official']
print('yes' if p2 > p1 + 0.0005 else 'no')")
if [ "$USE" != yes ]; then
  { echo "pass 2 NOT used: it did not beat pass 1 by more than 0.0005 on the report half"; grep -h "RESULT\|pass 2 minus" "$R/logs/night_pass2_fit.log" | tail -3
    echo "the final files stay output_quick/"; } | tee "$R/NIGHT_RESULT.txt"
  echo "total $(( ($(date +%s) - T0) / 60 )) min"; exit 0
fi
cp -f "$MIR/models/pass2.txt" "$MIR/models/decision_pass2.json" "$R/models/"
step pass2_apply     "$N/pred_pass2/_DONE"               env ER_ROOT="$R" ER_PASS1="$N/pred_na" python -u pass2.py apply
step pass2_write     "$R/output_pass2/matching_results.tsv" env ER_ROOT="$R" ER_DECISION=decision_pass2.json ER_PRED="$N/pred_pass2" ER_OUT="$R/output_pass2" python -u predict.py write
python utils/validate_submission.py --matching "$R/output_pass2/matching_results.tsv" --candidate "$R/output_pass2/candidate_pairs.tsv" \
  --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" | tee "$R/output_pass2/validator.log"
{ echo "pass 2 USED (beat pass 1 by more than 0.0005 on the held-out report half):"; grep -h "RESULT\|pass 2 minus" "$R/logs/night_pass2_fit.log" | tail -3; } | tee "$R/output_pass2/RESULT.txt"
tar czf "$R/output_pass2/logs.tgz" run_night.out $(ls logs/night_*.log)
echo "total $(( ($(date +%s) - T0) / 60 )) min. Final files: output_pass2/"
