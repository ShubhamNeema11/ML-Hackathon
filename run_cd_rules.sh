#!/usr/bin/env bash
# Generic models (ranker_cd, name-only noaddr_cdrq with the raw-name reranker, pass 2) applied with the country-specific normalization
# rules switched back ON at test time. No training: the rules only change features of records from a country the training data does not
# contain, so the trained models are unchanged. The generic-run test files are kept as *_gen. -> output_cdr_rules (rule of output_final)
# and output_cdr_rules_x (exact-metric rule).
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_CHUNK=50000 ER_NA_TFIDF=1 ER_CODE_UNSEEN=1 ER_FR_LEGAL=1 ER_FR_ADDR=1 ER_FR_NAME=1 ER_FR_GENERIC=1
N=normalized
B=(env ER_CE=1 ER_CE_TAG=_v2 ER_FEAT_TAG=_ce2)
RR=(env ER_CE_TAG=_nar ER_CE_TEXT=raw_name ER_CE_MODEL_DIR=ce_nar ER_CE_BASE=intfloat/multilingual-e5-small ER_CE_MAXLEN=48 ER_CE_DTYPE=fp16 ER_CE_SCORE_BATCH=512)
step() { local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "logs/rules_$name.log" 2>&1; then echo "[$name] FAILED:"; tail -n 15 "logs/rules_$name.log"; exit 1; fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"; }
if [ ! -f "$N/.rules_on" ]; then   # keep the generic-run test files
  for f in feat_noaddr2_test feat_noaddr2r_test feat_noaddr2rq_test na_rrpairs_test ce_test_nar; do [ -f "$N/$f.parquet" ] && mv "$N/$f.parquet" "$N/${f}_gen.parquet"; done
  touch "$N/.rules_on"
fi
step score    "$N/pred_cd_fr/_DONE" "${B[@]}" ER_MODEL=ranker_cd.txt ER_PRED="$N/pred_cd_fr" python -u predict.py score
step na_feat  "$N/feat_noaddr2_test.parquet" python -u noaddr.py features test
step raw      "$N/feat_noaddr2r_test.parquet" env ER_NA_RAW=1 python -u noaddr.py raw test
step rr_pairs "$N/na_rrpairs_test.parquet" env ER_NA_RAW=1 python -u noaddr.py rr_pairs test
step rr_score "$N/ce_test_nar.parquet" "${RR[@]}" ER_CE_PAIRS="$N/na_rrpairs_test.parquet" python -u crossenc.py score test
step rr_join  "$N/feat_noaddr2rq_test.parquet" env ER_NA_RAW=1 ER_NA_RR=1 python -u noaddr.py rr_join test
step na_apply "$N/pred_na_cdrq_fr/_DONE" env ER_NA_RAW=1 ER_NA_RR=1 ER_NOADDR_MODEL=noaddr_cdrq.txt ER_PRED_IN="$N/pred_cd_fr" ER_PRED_OUT="$N/pred_na_cdrq_fr" python -u noaddr.py apply
step p2_apply "$N/pred_p2_cdrq_fr/_DONE" env ER_PASS1="$N/pred_na_cdrq_fr" ER_PASS2_OUT="$N/pred_p2_cdrq_fr" python -u pass2.py apply
for spec in "output_cdr_rules decision_pass2.json" "output_cdr_rules_x decision_pass2_exact.json"; do
  set -- $spec
  step "write_$1" "$1/matching_results.tsv" env ER_DECISION=$2 ER_PRED="$N/pred_p2_cdrq_fr" ER_OUT=$1 python -u predict.py write
  python utils/validate_submission.py --matching $1/matching_results.tsv --candidate $1/candidate_pairs.tsv --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > $1/validator.log 2>&1
  echo "$1 ($2): $(tail -1 $1/validator.log)"
done
python - <<'PY'
import polars as pl
def pairs(p):
    x = pl.read_csv(p + "/matching_results.tsv", separator="\t", infer_schema_length=0, quote_char=None).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return x.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
F = pairs("output_final")
for o in ("output_cdr_rules", "output_cdr_rules_x"):
    b = pairs(o); print(f"{o}: {b.height:,} pairs (output_final {F.height:,}); only in final {F.join(b, on=['source1_entity_id','m'], how='anti').height:,}; only in {o} {b.join(F, on=['source1_entity_id','m'], how='anti').height:,}")
PY
echo "RULES DONE $(date +%H:%M:%S)"
