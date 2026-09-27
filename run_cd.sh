#!/usr/bin/env bash
# GENERIC (country-agnostic) model with CODE DROPOUT: the state / legal code features are kept but blanked on 50% of the training rows,
# unseen code values become unknown at prediction; no hand-written French rules. Nothing is learned from test data. logs/cd_*.log
#   bash run_cd.sh      -> output_cd/ (no pass 2) and output_cd_p2/ (with pass 2), both validated; logs/cd_gate.txt
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_CHUNK=50000 ER_NA_TFIDF=1
export ER_CODE_DROPOUT=0.5 ER_CODE_UNSEEN=1 ER_FR_LEGAL=0 ER_FR_ADDR=0 ER_FR_NAME=0 ER_FR_GENERIC=0   # country-agnostic: code dropout 50% (loco: unseen +0.0038, seen +0.0004), unseen codes -> unknown, no French rules
N=normalized; mkdir -p logs
B=(env ER_CE=1 ER_CE_TAG=_v2 ER_FEAT_TAG=_ce2)
NA=(env ER_NOADDR_MODEL=noaddr_cd.txt ER_B2=ranker_cd.txt ER_FEAT_TAG=_ce2)
step() { local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "logs/cd_$name.log" 2>&1; then echo "[$name] FAILED:"; tail -n 15 "logs/cd_$name.log"; exit 1; fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"; }
# 1. ranker without codes (same features files; the codes are just not used)
step fit_cd   "models/decision_cd.json"   "${B[@]}" ER_MODEL=ranker_cd.txt ER_DECISION=decision_cd.json python -u ranker.py fit
python -c "import json; b=json.load(open('models/decision_b2.json'))['held']; m=json.load(open('models/decision_cd.json'))['held']; print('ranker alone (held-out, report / tuned half): B2', round(b['early-stop half']['official'],4), '/', round(b['thr-half']['official'],4), '-> generic', round(m['early-stop half']['official'],4), '/', round(m['thr-half']['official'],4))" | tee logs/cd_gate.txt
# 2. specialist without codes (tuned settings), joint decision
step na_fit   "models/noaddr_cd.txt"      "${NA[@]}" ER_NA_LR=0.05 ER_NA_LEAVES=127 ER_NA_MINLEAF=400 ER_NA_ROUNDS=9000 python -u noaddr.py fit
step na_joint "models/decision_na_cd.json" "${NA[@]}" ER_DECISION=decision_na_cd.json python -u noaddr.py joint
python -c "import json; c=json.load(open('models/decision_na.json'))['held']; n=json.load(open('models/decision_na_cd.json'))['held']; print('whole held-out (report / tuned half): current', round(c['early-stop half']['official'],4), '/', round(c['thr-half']['official'],4), '-> generic', round(n['early-stop half']['official'],4), '/', round(n['thr-half']['official'],4))" | tee -a logs/cd_gate.txt
# 3. test scoring (French rules off)
step score    "$N/pred_cd/_DONE"          "${B[@]}" ER_MODEL=ranker_cd.txt ER_PRED="$N/pred_cd" python -u predict.py score
# 4. no-address test features without French rules (the France-rule version is kept as feat_noaddr2_test_fr3.parquet), specialist, pass 2, files
[ -f "$N/feat_noaddr2_test_fr3.parquet" ] || cp -p "$N/feat_noaddr2_test.parquet" "$N/feat_noaddr2_test_fr3.parquet"
[ -f "$N/.feat_noaddr2_test_cd" ] || { rm -f "$N/feat_noaddr2_test.parquet"; touch "$N/.feat_noaddr2_test_cd"; }
step na_feat  "$N/feat_noaddr2_test.parquet" python -u noaddr.py features test
step na_apply "$N/pred_na_cd/_DONE"       "${NA[@]}" ER_PRED_IN="$N/pred_cd" ER_PRED_OUT="$N/pred_na_cd" python -u noaddr.py apply
step write    "output_cd/matching_results.tsv" env ER_DECISION=decision_na_cd.json ER_PRED="$N/pred_na_cd" ER_OUT=output_cd python -u predict.py write
step p2_apply "$N/pred_p2_cd/_DONE"       env ER_PASS1="$N/pred_na_cd" ER_PASS2_OUT="$N/pred_p2_cd" python -u pass2.py apply
step p2_write "output_cd_p2/matching_results.tsv" env ER_DECISION=decision_pass2.json ER_PRED="$N/pred_p2_cd" ER_OUT=output_cd_p2 python -u predict.py write
for o in output_cd output_cd_p2; do
  python utils/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir "$(python -c 'from block import DATASET; print(DATASET / "test")')" > $o/validator.log 2>&1
  echo "$o: $(tail -1 $o/validator.log)"; cp logs/cd_gate.txt $o/GATE.txt
done
echo "$(date +%H:%M:%S) GENERIC DONE"
