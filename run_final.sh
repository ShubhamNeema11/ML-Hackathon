#!/usr/bin/env bash
# Final file: pass 2 (fitted on the training mirror, +0.0006 on 766k held-out owned records) applied on top of the France3 scores
# (B2 + all French fixes + no-address specialist) -> output_final/ (+ validator, RESULT.txt). Waits for France3's scores.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_CHUNK=50000
N=normalized
until [ -f "$N/pred_na_fr3/_DONE" ] || grep -q "FAILED" run_france3.out 2>/dev/null; do sleep 30; done
[ -f "$N/pred_na_fr3/_DONE" ] || { echo "France3 failed: see run_france3.out"; exit 1; }
echo "$(date +%H:%M:%S) pass-2 apply on pred_na_fr3"
[ -f "$N/pred_pass2_fr3/_DONE" ] || ER_PASS1="$N/pred_na_fr3" ER_PASS2_OUT="$N/pred_pass2_fr3" python -u pass2.py apply > logs/final_pass2_apply.log 2>&1 || { echo "pass2 apply FAILED"; tail -15 logs/final_pass2_apply.log; exit 1; }
echo "$(date +%H:%M:%S) write"
[ -f output_final/matching_results.tsv ] || ER_DECISION=decision_pass2.json ER_PRED="$N/pred_pass2_fr3" ER_OUT=output_final python -u predict.py write > logs/final_write.log 2>&1 || { echo "write FAILED"; tail -15 logs/final_write.log; exit 1; }
python utils/validate_submission.py --matching output_final/matching_results.tsv --candidate output_final/candidate_pairs.tsv \
  --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > output_final/validator.log 2>&1; tail -1 output_final/validator.log
python - <<'PY' | tee output_final/RESULT.txt
import json, polars as pl
d = json.load(open("models/decision_pass2.json"))
print("pipeline: B2 (address records) + no-address specialist (all 309,844 no-address training records, tuned) + French fixes")
print("          (legal-form codes, address canonicalization, legal forms in names, hand-written French generic words) + pass 2 (sibling support)")
print(f"pass 2 on the training mirror, held-out report half (766k owned records): pass 1 {d['pass1_held']['report-half']['official']:.4f} -> pass 2 {d['held']['report-half']['official']:.4f}")
print("decision rule (pass 2):", {k: d[k] for k in ("thr_addr", "thr_noaddr", "margin")})
def pairs(p):
    x = pl.read_csv(p, separator="\t", infer_schema_length=0, quote_char=None).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return x.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
a, b = pairs("output_best/matching_results.tsv"), pairs("output_final/matching_results.tsv")
print(f"assigned pairs: output_best (LB 0.98011) {a.height:,} -> output_final {b.height:,}; only in best {a.join(b, on=['source1_entity_id', 'm'], how='anti').height:,}, only in final {b.join(a, on=['source1_entity_id', 'm'], how='anti').height:,}")
PY
echo "FINAL DONE $(date +%H:%M:%S)"
