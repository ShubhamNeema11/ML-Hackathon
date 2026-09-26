#!/usr/bin/env bash
# QUICK path: keep everything that is already trained and scored, train ONLY the no-address specialist on all ~310k no-address training
# records, patch the existing B2 test scores for records without an address, write the final files.
#
# SageMaker JupyterLab (defaults: work folder ~/er_work, dataset ~/ML-Hackathon/data):
#   laptop:     python stage_upload.py  -> Downloads/er_work_upload/er_work; S3 console: bucket -> Upload -> Add folder -> that er_work folder
#               (or, with the AWS CLI: bash upload_quick.sh <bucket>)             about 7.75 GB either way; bucket name should start with "sagemaker-"
#   SageMaker:  cd ~/ML-Hackathon && git pull
#               aws s3 sync s3://<bucket>/er_work ~/er_work --only-show-errors     (or let this script do it: S3_IN=s3://<bucket> bash run_quick.sh)
#               bash aws/sagemaker_setup.sh                                        (must print READY)
#               S3_IN=s3://<bucket> nohup bash run_quick.sh > run_quick.out 2>&1 &     then: tail -f run_quick.out
# On the laptop itself: bash run_quick.sh   (work folder = the repository). Resumable: finished steps are skipped.
#
# Reused as they are: normalized data, sparse + dense blocking, e5 embedder, cross-encoder ce_er2 and its test scores, ranker B2
# (models/ranker_b2.txt) and its test scores normalized/pred_final. Nothing of the address pipeline is retrained or re-scored.
# Output: $ER_ROOT/output_quick/ : matching_results.tsv, candidate_pairs.tsv, validator.log, RESULT.txt (held-out numbers), logs.tgz.
# The previous specialist (31k records) is kept as models/noaddr_c_31k.txt; final_submission/ is never touched.
set -uo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
# work folder: $ER_ROOT, else the repository when it holds normalized/ (laptop), else ~/er_work (SageMaker)
if [ -z "${ER_ROOT:-}" ]; then [ -d "$PWD/normalized/pred_final" ] && ER_ROOT="$PWD" || ER_ROOT="$HOME/er_work"; fi
# dataset (train_ground_truth.tsv is read by several steps): $ER_DATASET, else <repo>/data (SageMaker), else block.py's default
if [ -z "${ER_DATASET:-}" ] && [ -d "$PWD/data/train" ]; then ER_DATASET="$PWD/data"; fi
[ -n "${ER_DATASET:-}" ] && export ER_DATASET
export PYTHONUTF8=1 ER_ROOT
echo "work folder: $ER_ROOT   dataset: ${ER_DATASET:-(block.py default)}"
if [ -n "${S3_IN:-}" ]; then   # the uploaded er_work folder (s3://<bucket>/er_work) -> $ER_ROOT; only missing / changed files are copied
  echo "copying ${S3_IN%/}/er_work -> $ER_ROOT"; mkdir -p "$ER_ROOT"
  aws s3 sync "${S3_IN%/}/er_work" "$ER_ROOT" --only-show-errors || { echo "aws s3 sync failed (AccessDenied: the SageMaker role cannot read this bucket; use a bucket named sagemaker-...)"; exit 1; }
fi
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "torch cannot see a GPU (the dense search and the TF-IDF search need one)"; exit 1; }
python -c "from block import DATASET; import sys; p = DATASET / 'train' / 'train_ground_truth.tsv'; sys.exit(0 if p.exists() else print('missing', p) or 1)" || exit 1
export ER_NA_TFIDF=1 ER_EXTRAS=0 ER_B2=ranker_b2.txt ER_FEAT_TAG=_ce2 ER_DECISION=decision_na.json
N="$ER_ROOT/normalized"; M="$ER_ROOT/models"; OUT="$ER_ROOT/output_quick"
mkdir -p "$ER_ROOT/logs"

for f in "$N/pred_final/_DONE" "$N/cand/test_sparse/_DONE" "$N/cand/test_dense/_DONE" "$N/feat_eval_ce2.parquet" "$N/eval_queries.parquet" "$N/test_ids.parquet"          "$N/source1.parquet" "$N/source2.parquet" "$N/source3.parquet" "$N/test_source1.parquet" "$N/test_source2.parquet" "$N/test_source3.parquet"          "$N/state_fix_source1.parquet" "$N/state_fix_test_source1.parquet" "$M/ranker_b2.txt" "$M/struct_vocab.json" "$M/e5_er/model.safetensors"; do
  [ -e "$f" ] || { echo "missing $f: upload it (bash upload_quick.sh <bucket> on the laptop, then aws s3 sync on SageMaker)"; exit 1; }
done
# the old 31k-record specialist and its decision rule are kept aside once, so the steps below train / tune them again
[ -f "$M/noaddr_c.txt" ] && [ ! -f "$M/noaddr_c_31k.txt" ] && mv "$M/noaddr_c.txt" "$M/noaddr_c_31k.txt" && mv -f "$M/decision_na.json" "$M/decision_na_31k.json" 2>/dev/null

T0=$(date +%s)
step() {   # step <name> <done-file> <command...>
  local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "$ER_ROOT/logs/quick_$name.log" 2>&1; then
    echo "[$name] FAILED, last lines of logs/quick_$name.log:"; tail -n 15 "$ER_ROOT/logs/quick_$name.log"; exit 1
  fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"
}

# 1. training data for the specialist: all no-address training records (held-out entities, evaluation queries, pass-2 orphan sample excluded)
step trainall_queries "$N/trainall_queries.parquet"                python -u noaddr.py trainall_queries
step trainall_sparse  "$N/cand/trainall_sparse/_DONE"              python -u noaddr.py regular_sparse
step trainall_dense   "$N/cand/trainall_dense/_DONE"               python -u noaddr.py regular_dense
step trainall_tfidf   "$N/cand/na_tfidf_trainall.parquet"          python -u blocking_noaddr.py build trainall
step trainall_feat    "$N/feat_noaddr2_trainall.parquet"           python -u noaddr.py features trainall
# 2. held-out candidates / features (built earlier on the laptop; made again only if missing)
step eval_tfidf       "$N/cand/na_tfidf_eval.parquet"              python -u blocking_noaddr.py build eval
step eval_feat        "$N/feat_noaddr2_eval.parquet"               python -u noaddr.py features eval
# 3. test candidates / features of the 265k test records without an address
step test_tfidf       "$N/cand/na_tfidf_test.parquet"              python -u blocking_noaddr.py build test
step test_feat        "$N/feat_noaddr2_test.parquet"               python -u noaddr.py features test
# 4. train the specialist, tune the joint decision rule (B2 for address records + specialist) on the held-out set
step fit              "$M/noaddr_c.txt"                            python -u noaddr.py fit
step joint            "$M/decision_na.json"                        python -u noaddr.py joint
# 5. patch B2's test scores for the records without an address, write and validate
step apply            "$N/pred_na/_DONE"                           env ER_PRED_IN="$N/pred_final" ER_PRED_OUT="$N/pred_na" python -u noaddr.py apply
step write            "$OUT/matching_results.tsv"                  env ER_PRED="$N/pred_na" ER_OUT="$OUT" python -u predict.py write
VAL=""; for c in "$ER_ROOT/utils/validate_submission.py" "$PWD/utils/validate_submission.py"; do [ -f "$c" ] && VAL=$c && break; done
TEST_DIR="${ER_DATASET:-}/test"; [ -d "$TEST_DIR" ] || TEST_DIR=$(python -c "from block import DATASET; print(DATASET / 'test')")
[ -n "$VAL" ] && python "$VAL" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$TEST_DIR" | tee "$OUT/validator.log"

echo
{ echo "held-out expected official score (tuned half / report half):"; grep -h "RESULT" "$ER_ROOT/logs/quick_joint.log" | tail -2
  echo; echo "specialist on the no-address group:"; grep -h -A2 "orphan weight" "$ER_ROOT/logs/quick_fit.log" | tail -3; } | tee "$OUT/RESULT.txt"
tar czf "$OUT/logs.tgz" -C "$ER_ROOT" $(cd "$ER_ROOT" && ls logs/quick_*.log)
echo "total $(( ($(date +%s) - T0) / 60 )) min. Final files: $OUT/  (download matching_results.tsv, candidate_pairs.tsv, RESULT.txt, logs.tgz)"
