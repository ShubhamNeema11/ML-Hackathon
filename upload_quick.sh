#!/usr/bin/env bash
# Laptop side (Git Bash): upload EXACTLY what run_quick.sh needs on SageMaker (about 7.3 GB), nothing else.
#
#   bash upload_quick.sh <bucket>          e.g.  bash upload_quick.sh sagemaker-er-622532143565
#
# The bucket must be readable by the SageMaker execution role: a name starting with "sagemaker-" is (default SageMaker policy);
# any other bucket needs s3:GetObject / s3:ListBucket for that role. The bucket is created when it does not exist.
# Needs the AWS CLI logged in to the SAME account as SageMaker (aws configure). Code comes from GitHub (git pull), the raw dataset is already on the space.
#
# Uploaded to s3://<bucket>/er_work/ (then on SageMaker:  aws s3 sync s3://<bucket>/er_work ~/er_work --only-show-errors):
#   normalized/  source1-3, test_source1-3, state_fix_* (corrected states), eval_queries, test_ids, feat_eval_ce2 (B2's held-out features),
#                feat_noaddr2_eval + cand/na_tfidf_eval (held-out no-address features), cand/test_sparse + cand/test_dense (test blocking),
#                pred_final (B2's scores of every test pair)
#   models/      e5_er (embedder), ranker_b2.txt, decision_b2.json, struct_vocab.json
set -euo pipefail
cd "$(dirname "$0")"
BUCKET="${1:?usage: bash upload_quick.sh <bucket>}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-2}"
aws sts get-caller-identity --query Account --output text | xargs echo "AWS account:" || { echo "AWS CLI not logged in: run aws configure"; exit 1; }
aws s3 ls "s3://$BUCKET" >/dev/null 2>&1 || aws s3 mb "s3://$BUCKET"
D="s3://$BUCKET/er_work"

FILES=(normalized/source1.parquet normalized/source2.parquet normalized/source3.parquet
       normalized/test_source1.parquet normalized/test_source2.parquet normalized/test_source3.parquet
       normalized/state_fix_source1.parquet normalized/state_fix_source2.parquet normalized/state_fix_source3.parquet
       normalized/state_fix_test_source1.parquet normalized/state_fix_test_source2.parquet normalized/state_fix_test_source3.parquet
       normalized/eval_queries.parquet normalized/test_ids.parquet normalized/feat_eval_ce2.parquet
       normalized/feat_noaddr2_eval.parquet normalized/cand/na_tfidf_eval.parquet
       models/ranker_b2.txt models/decision_b2.json models/struct_vocab.json)
DIRS=(normalized/cand/test_sparse normalized/cand/test_dense normalized/pred_final models/e5_er)

for f in "${FILES[@]}" "${DIRS[@]}"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f normalized/pred_final/_DONE ] || { echo "normalized/pred_final is incomplete (no _DONE)"; exit 1; }
for f in "${FILES[@]}"; do echo "  $f"; aws s3 cp "$f" "$D/$f" --only-show-errors; done
for d in "${DIRS[@]}"; do echo "  $d/"; aws s3 sync "$d" "$D/$d" --exclude "ckpt/*" --only-show-errors; done
echo
aws s3 ls "$D/" --recursive --summarize | tail -2
echo "done. On SageMaker:  aws s3 sync $D ~/er_work --only-show-errors"
