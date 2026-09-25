#!/usr/bin/env bash
# One-time setup on an AWS GPU instance (Ubuntu with NVIDIA drivers, e.g. the "Deep Learning AMI (Ubuntu)").
# Run from the repository root:   bash aws/setup.sh
# Optional: S3_DATASET=s3://your-bucket/dataset bash aws/setup.sh    (copies the challenge data with the AWS CLI)
set -euo pipefail

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# the pipeline was tested with Python 3.11; Ubuntu 22.04 ships 3.10, so install 3.11 (deadsnakes) when it is missing
PY=python3.11
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "== installing Python 3.11"
  sudo apt-get update -y
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -y
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi
echo "== python"; "$PY" --version

"$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# the CUDA 12.4 build of torch first, then everything else pinned in requirements.txt
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

mkdir -p /data/er_work
if [ -n "${S3_DATASET:-}" ]; then
  echo "== dataset from $S3_DATASET"; mkdir -p /data/dataset
  aws s3 sync "$S3_DATASET" /data/dataset --only-show-errors
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
cat <<'MSG'

Setup done. Start the pipeline inside tmux so it survives a dropped connection:

  tmux new -s er
  source .venv/bin/activate
  python main.py --dataset /data/dataset --root /data/er_work        # dataset/ must contain train/ and test/
  # detach: Ctrl-b d      re-attach: tmux attach -t er      progress: tail -f /data/er_work/logs/*.log

Interrupted or failed? Fix the cause and run the same command again: finished steps are skipped.
MSG
