#!/usr/bin/env bash
# Split decision rule: exact-metric rule (lower thresholds) for records from countries present in the training data, the stricter rule of
# the 0.9815 file for records from countries the training data does not contain. Waits for run_cd_rules.sh.
cd "$(dirname "$0")"
export PYTHONUTF8=1
until grep -q "RULES DONE\|FAILED" run_cd_rules.out 2>/dev/null; do sleep 30; done
N=normalized
for spec in "output_cd_p2rq_split pred_p2_cdrq" "output_cdr_rules_split pred_p2_cdrq_fr"; do
  set -- $spec
  [ -d "$N/$2" ] || { echo "$2 missing"; continue; }
  [ -f "$1/matching_results.tsv" ] || ER_DECISION=decision_pass2_exact.json ER_DECISION_UNSEEN=decision_pass2.json ER_PRED="$N/$2" ER_OUT=$1 python -u predict.py write > logs/split_$1.log 2>&1 || { echo "$1 write FAILED"; continue; }
  python utils/validate_submission.py --matching $1/matching_results.tsv --candidate $1/candidate_pairs.tsv --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > $1/validator.log 2>&1
  echo "$1: $(tail -1 $1/validator.log)"
done
python - <<'PY'
import polars as pl
def pairs(p):
    x = pl.read_csv(p + "/matching_results.tsv", separator="\t", infer_schema_length=0, quote_char=None).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return x.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
A = pairs("output_cd_p2rq")
for o in ("output_cd_p2rq_split", "output_cdr_rules", "output_cdr_rules_split"):
    try: b = pairs(o)
    except Exception: continue
    print(f"{o}: {b.height:,} pairs vs the 0.9815 file {A.height:,}; removed {A.join(b, on=['source1_entity_id','m'], how='anti').height:,}, added {b.join(A, on=['source1_entity_id','m'], how='anti').height:,}")
PY
echo "SPLIT DONE $(date +%H:%M:%S)"
