#!/usr/bin/env bash
# Final file with the raw-name specialist: generic scores (pred_cd) -> no-address rows re-scored by the raw-name specialist (noaddr_cdr,
# +0.0003 on held-out) -> pass 2 -> exact-metric-tuned pass-2 rule -> output_cd_p2xr/ (+ validator). Waits for run_cd.sh.
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_NA_TFIDF=1 ER_NA_RAW=1 ER_CODE_UNSEEN=1 ER_FR_LEGAL=0 ER_FR_ADDR=0 ER_FR_NAME=0 ER_FR_GENERIC=0
until grep -q "GENERIC DONE\|FAILED" run_cd.out 2>/dev/null; do sleep 20; done
N=normalized
[ -f "$N/feat_noaddr2r_test.parquet" ] || python -u noaddr.py raw test > logs/final_raw_feat.log 2>&1 || { echo "raw features FAILED"; exit 1; }
[ -f "$N/pred_na_cdr/_DONE" ] || ER_NOADDR_MODEL=noaddr_cdr.txt ER_PRED_IN="$N/pred_cd" ER_PRED_OUT="$N/pred_na_cdr" python -u noaddr.py apply > logs/final_raw_apply.log 2>&1 || { echo "apply FAILED"; exit 1; }
[ -f "$N/pred_p2_cdr/_DONE" ] || ER_PASS1="$N/pred_na_cdr" ER_PASS2_OUT="$N/pred_p2_cdr" python -u pass2.py apply > logs/final_raw_p2.log 2>&1 || { echo "pass2 FAILED"; exit 1; }
[ -f output_cd_p2xr/matching_results.tsv ] || ER_DECISION=decision_pass2_exact.json ER_PRED="$N/pred_p2_cdr" ER_OUT=output_cd_p2xr python -u predict.py write > logs/final_raw_write.log 2>&1 || { echo "write FAILED"; exit 1; }
python utils/validate_submission.py --matching output_cd_p2xr/matching_results.tsv --candidate output_cd_p2xr/candidate_pairs.tsv --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > output_cd_p2xr/validator.log 2>&1
echo "output_cd_p2xr: $(tail -1 output_cd_p2xr/validator.log)"
echo "FINAL RAW DONE $(date +%H:%M:%S)"
