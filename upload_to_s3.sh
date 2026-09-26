#!/usr/bin/env bash
# One command from Git Bash on your laptop: packages the code and copies everything SageMaker Studio needs to S3.
#   bash upload_to_s3.sh                      -> code + raw dataset + normalized/ + models/   (continue from where we are)
#   MODE=fresh bash upload_to_s3.sh           -> code + raw dataset only (Studio recomputes everything, 4-6 h)
#   bash upload_to_s3.sh my-bucket-name       -> another bucket (the name must start with "sagemaker-")
set -euo pipefail
cd "$(dirname "$0")"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-2}"
BUCKET="${1:-sagemaker-er-050307929098}"
SRC="${DATASET_PARENT:-/c/Users/Lenovo/Downloads/6ab10eb3b23ba_student_resource/student_resource}"
MODE="${MODE:-continue}"

aws sts get-caller-identity --query Arn --output text > /dev/null || { echo "AWS login is not working: run 'aws configure' first"; exit 1; }
aws s3 mb "s3://$BUCKET" 2>/dev/null || true

echo "== code (working folder, without data / models / outputs)"
tar czf er_code.tgz --exclude=normalized --exclude=models --exclude=models_B1 --exclude=output_submitted_0957 \
  --exclude=final_submission --exclude=final_overnight --exclude=final_overnight_v3 --exclude=interim_submission \
  --exclude=logs --exclude=__pycache__ --exclude=.git --exclude='*.log' --exclude='*.tgz' .
aws s3 cp er_code.tgz "s3://$BUCKET/er_code.tgz" --only-show-errors

echo "== raw dataset + validator"
aws s3 sync "$SRC/dataset" "s3://$BUCKET/dataset" --only-show-errors
aws s3 sync "$SRC/utils" "s3://$BUCKET/utils" --only-show-errors

if [ "$MODE" = "continue" ]; then
  echo "== normalized data, candidates, features and models (about 9 GB; the scored test pairs are left out)"
  aws s3 sync normalized "s3://$BUCKET/er_work/normalized" --exclude "pred_*" --only-show-errors
  aws s3 sync models "s3://$BUCKET/er_work/models" --only-show-errors
fi
echo "done. In SageMaker Studio: upload sagemaker_run.ipynb and choose Run All."
