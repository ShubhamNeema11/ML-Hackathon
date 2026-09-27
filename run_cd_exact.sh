#!/usr/bin/env bash
# After run_cd.sh: the same generic scores, written with the decision rules tuned on the EXACT competition metric (exact_tune.py, training
# mirror, held-out S1 entities, untouched half): pass 2 0.9887 -> 0.9891, pass 1 0.9884 -> 0.9888.
cd "$(dirname "$0")"
export PYTHONUTF8=1
until grep -q "GENERIC DONE\|FAILED" run_cd.out 2>/dev/null; do sleep 30; done
grep -q "GENERIC DONE" run_cd.out || { echo "run_cd.sh failed"; exit 1; }
for spec in "output_cd_p2x decision_pass2_exact.json normalized/pred_p2_cd" "output_cdx decision_exact.json normalized/pred_na_cd"; do
  set -- $spec
  [ -f "$1/matching_results.tsv" ] || ER_DECISION=$2 ER_PRED=$3 ER_OUT=$1 python -u predict.py write > "logs/cdx_$1.log" 2>&1
  python utils/validate_submission.py --matching $1/matching_results.tsv --candidate $1/candidate_pairs.tsv --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > $1/validator.log 2>&1
  echo "$1 ($2): $(tail -1 $1/validator.log)"
done
echo "EXACT DONE $(date +%H:%M:%S)"
