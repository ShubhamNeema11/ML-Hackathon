#!/usr/bin/env bash
# Final file with the raw-name reranker in the name-only pipeline: generic scores (pred_cd) -> top-8 candidates per no-address record
# (noaddr_cdr) scored by the raw-name reranker (ce_nar) -> specialist with reranker features (noaddr_cdrq, held-out +0.0002 over cdr at
# the same rule) -> pass 2 -> exact-metric-tuned pass-2 rule -> output_cd_p2xrq/ (+ validator). Test data is only scored, never trained on.
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_NA_TFIDF=1 ER_NA_RAW=1 ER_CODE_UNSEEN=1 ER_FR_LEGAL=0 ER_FR_ADDR=0 ER_FR_NAME=0 ER_FR_GENERIC=0
RR=(env ER_CE_TAG=_nar ER_CE_TEXT=raw_name ER_CE_MODEL_DIR=ce_nar ER_CE_BASE=intfloat/multilingual-e5-small ER_CE_MAXLEN=48 ER_CE_DTYPE=fp16 ER_CE_SCORE_BATCH=512)
N=normalized
[ -f "$N/na_rrpairs_test.parquet" ] || python -u noaddr.py rr_pairs test > logs/rq_pairs.log 2>&1 || { echo "rr_pairs FAILED"; exit 1; }
[ -f "$N/ce_test_nar.parquet" ] || "${RR[@]}" ER_CE_PAIRS="$N/na_rrpairs_test.parquet" python -u crossenc.py score test > logs/rq_score.log 2>&1 || { echo "score FAILED"; exit 1; }
[ -f "$N/feat_noaddr2rq_test.parquet" ] || ER_NA_RR=1 python -u noaddr.py rr_join test > logs/rq_join.log 2>&1 || { echo "rr_join FAILED"; exit 1; }
[ -f "$N/pred_na_cdrq/_DONE" ] || ER_NA_RR=1 ER_NOADDR_MODEL=noaddr_cdrq.txt ER_PRED_IN="$N/pred_cd" ER_PRED_OUT="$N/pred_na_cdrq" python -u noaddr.py apply > logs/rq_apply.log 2>&1 || { echo "apply FAILED"; exit 1; }
[ -f "$N/pred_p2_cdrq/_DONE" ] || ER_PASS1="$N/pred_na_cdrq" ER_PASS2_OUT="$N/pred_p2_cdrq" python -u pass2.py apply > logs/rq_p2.log 2>&1 || { echo "pass2 FAILED"; exit 1; }
[ -f output_cd_p2xrq/matching_results.tsv ] || ER_DECISION=decision_pass2_exact.json ER_PRED="$N/pred_p2_cdrq" ER_OUT=output_cd_p2xrq python -u predict.py write > logs/rq_write.log 2>&1 || { echo "write FAILED"; exit 1; }
python utils/validate_submission.py --matching output_cd_p2xrq/matching_results.tsv --candidate output_cd_p2xrq/candidate_pairs.tsv --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > output_cd_p2xrq/validator.log 2>&1
echo "output_cd_p2xrq: $(tail -1 output_cd_p2xrq/validator.log)"
echo "FINAL RQ DONE $(date +%H:%M:%S)"
