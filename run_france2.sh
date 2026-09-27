#!/usr/bin/env bash
# France legal-form fix + French address canonicalization (ranker._fr_addr): same steps as run_france.sh, into output_france2/.
# specialist, write output_france2/ (+ validator) and compare the French uncertain share before / after. Resumable; output_best/ is untouched.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_CHUNK=50000 ER_FR_LEGAL=1 ER_FR_ADDR=1
N=normalized; mkdir -p logs
step() { local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "logs/france2_$name.log" 2>&1; then echo "[$name] FAILED:"; tail -n 15 "logs/france2_$name.log"; exit 1; fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"; }
step score_b2 "$N/pred_final_fr2/_DONE" env ER_CE=1 ER_CE_TAG=_v2 ER_MODEL=ranker_b2.txt ER_PRED="$N/pred_final_fr2" python -u predict.py score
[ -f "$N/feat_noaddr2_test_prefr.parquet" ] || cp -p "$N/feat_noaddr2_test.parquet" "$N/feat_noaddr2_test_prefr.parquet"
[ -f "$N/.feat_noaddr2_test_fr2" ] || { rm -f "$N/feat_noaddr2_test.parquet"; touch "$N/.feat_noaddr2_test_fr2"; }
step na_feat "$N/feat_noaddr2_test.parquet" env ER_NA_TFIDF=1 ER_EXTRAS=0 python -u noaddr.py features test
step na_apply "$N/pred_na_fr2/_DONE" env ER_NA_TFIDF=1 ER_PRED_IN="$N/pred_final_fr2" ER_PRED_OUT="$N/pred_na_fr2" python -u noaddr.py apply
step write "output_france2/matching_results.tsv" env ER_DECISION=decision_na.json ER_PRED="$N/pred_na_fr2" ER_OUT=output_france2 python -u predict.py write
python utils/validate_submission.py --matching output_france2/matching_results.tsv --candidate output_france2/candidate_pairs.tsv \
  --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > output_france2/validator.log 2>&1; tail -1 output_france2/validator.log
python - <<'PY' | tee output_france2/FRANCE_CHECK.txt
import polars as pl
n1 = pl.scan_parquet("normalized/test_source1.parquet").select(pl.len()).collect().item()
q = pl.concat([pl.read_parquet(f"normalized/test_source{i}.parquet", columns=["country"]) for i in (2, 3)]).with_row_index("k").with_columns((pl.col("k") + n1).cast(pl.UInt32).alias("rec_i"))
for name, d in (("before (pred_na)", "normalized/pred_na"), ("after  (pred_na_fr2)", "normalized/pred_na_fr2")):
    t = q.join(pl.scan_parquet(f"{d}/part*.parquet").group_by("rec_i").agg(pl.col("p").max().alias("p1")).collect(), on="rec_i", how="left").with_columns(pl.col("p1").fill_null(0))
    s = t.group_by("country").agg(((pl.col("p1") >= 0.05) & (pl.col("p1") < 0.98)).mean().round(4).alias("unsure_0.05-0.98"), (pl.col("p1") >= 0.8).mean().round(4).alias("accepted_zone>=0.8")).sort("country")
    print(name, s.to_dicts())
def pairs(p):
    d = pl.read_csv(p, separator="\t", infer_schema_length=0, quote_char=None).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return d.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
a, b = pairs("output_best/matching_results.tsv"), pairs("output_france2/matching_results.tsv")
print("assigned pairs: output_best", a.height, " output_france2", b.height, " only in best", a.join(b, on=["source1_entity_id", "m"], how="anti").height, " only in france", b.join(a, on=["source1_entity_id", "m"], how="anti").height)
PY
echo "FRANCE DONE $(date +%H:%M:%S)"
