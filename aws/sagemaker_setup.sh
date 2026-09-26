#!/usr/bin/env bash
# One-time setup in a SageMaker JupyterLab terminal (no sudo needed). From the repository (~/ML-Hackathon):
#
#   git pull && bash aws/sagemaker_setup.sh
#
# It checks the GPU, the free disk, the dataset (~/ML-Hackathon/data/{train,test}) and the reused embedder, creates ONE virtualenv (.venv, about 8 GB:
# CUDA torch 2.6.0 + requirements), and verifies the imports. It changes nothing outside the repository folder, ~/er_work and pip's own files.
# Options:  ER_ROOT=$HOME/er_work   ER_DATASET=$HOME/ML-Hackathon/data   SKIP_E5=1 (embedder trained again; not what you want)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
ER_ROOT="${ER_ROOT:-$HOME/er_work}"; ER_DATASET="${ER_DATASET:-$REPO/data}"
ok()   { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
bad()  { echo "  [FAIL] $*"; FAILED=1; }
FAILED=0

echo "== GPU"
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then ok "GPU visible"; else bad "no GPU: switch the space to a GPU instance type (for example ml.g5.2xlarge)"; fi

echo "== disk (peak about 55 GB: 8 venv + 2.4 dataset + ~20 run files + ~10 models / model cache + ~12 pass-2 mirror, freed after pass 2)"
FREE=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
if [ "${FREE:-0}" -ge 70 ]; then ok "${FREE} GB free"; else bad "only ${FREE} GB free in $HOME: increase the space's storage (stop the space, edit the space settings)"; fi

echo "== dataset in $ER_DATASET"
for f in train/train_ground_truth.tsv train/train_source1.tsv train/train_source2.tsv train/train_source3.tsv test/test_source1.tsv test/test_source2.tsv test/test_source3.tsv; do
  if [ -s "$ER_DATASET/$f" ]; then ok "$f ($(du -h "$ER_DATASET/$f" | cut -f1))"; else bad "$f is missing or empty"; fi
done
[ -f "$REPO/utils/validate_submission.py" ] && ok "utils/validate_submission.py" || warn "utils/validate_submission.py missing (git pull): the last step would skip the validator"

echo "== reused embedder (models/e5_er)"
mkdir -p "$ER_ROOT/models" "$ER_ROOT/logs"
if [ -s "$ER_ROOT/models/e5_er/model.safetensors" ]; then
  ok "$ER_ROOT/models/e5_er is there"
else
  Z=""; for c in "$HOME/e5_er.zip" "$REPO/e5_er.zip" "$HOME/ML-Hackathon/e5_er.zip"; do [ -f "$c" ] && Z="$c" && break; done
  if [ -n "$Z" ]; then
    python3 -m zipfile -e "$Z" "$ER_ROOT/models/" && ok "unpacked $Z into $ER_ROOT/models/"
    [ -s "$ER_ROOT/models/e5_er/model.safetensors" ] || bad "the zip did not contain e5_er/model.safetensors"
  elif [ "${SKIP_E5:-0}" = 1 ]; then warn "SKIP_E5=1: the embedder will be fine-tuned again in the run (extra time)"
  else bad "e5_er.zip not found: make it on the laptop (python pack_e5.py), upload it with the JupyterLab upload button to your home folder, run this again"; fi
fi

echo "== python environment (.venv)"
python3 --version
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv || bad "could not create the virtualenv"; fi
if [ -x .venv/bin/python ]; then
  source .venv/bin/activate
  pip install --no-cache-dir --upgrade pip -q
  if ! python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.6.0') and torch.cuda.is_available() else 1)" 2>/dev/null; then
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || bad "torch install failed"
  fi
  grep -v '^torch==' requirements.txt > /tmp/req_no_torch.txt
  pip install --no-cache-dir -r /tmp/req_no_torch.txt || bad "requirements install failed"
  python - <<'PY' || FAILED=1
import torch, polars, lightgbm, sentence_transformers, transformers, rapidfuzz, sklearn
print("  torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("  polars", polars.__version__, "| lightgbm", lightgbm.__version__, "| sentence-transformers", sentence_transformers.__version__)
PY
  df -BG --output=avail "$HOME" | tail -1 | xargs echo "  free after installs (GB):"
fi

echo
if [ "$FAILED" = 0 ]; then
  echo "READY. Start the run (in the background, it survives a closed browser tab):"
  echo "  cd $REPO && nohup bash aws/run_full.sh > run_full.out 2>&1 &     then:  tail -f run_full.out"
else
  echo "NOT READY: fix the [FAIL] lines above and run this script again (it is safe to repeat)."
  exit 1
fi
