#!/usr/bin/env bash
# STAGE 1 (gate): fine-tune a bigger multilingual embedder (default BAAI/bge-m3) on the same training pairs as the current e5-small and
# measure the held-out owner recall of the dense channel, alone and united with the sparse channel. Nothing uses test data.
#
# SageMaker JupyterLab (GPU space, ml.g5.2xlarge or larger):
#   cd ~/ML-Hackathon && git pull && bash aws/sagemaker_setup.sh      (READY; set NEED_GB=20 if the space is small)
#   aws s3 sync s3://<bucket>/er_stage1 ~/er_stage1 --only-show-errors  (the folder made by stage1_upload.py on the laptop)
#   nohup bash aws/stage1_embed.sh > stage1.out 2>&1 &    then: tail -f stage1.out
#   (S3_OUT=s3://<bucket>/stage1_results also copies the results folder to S3)
# Output: ~/er_stage1/stage1_results/  RESULT.txt (recall + verdict), stage1_train.log, stage1_eval.log, m3_er_model.tgz (the fine-tuned embedder)
# Gate: the current fine-tuned e5-small reaches dense top-10 recall 0.9907 and union (sparse + dense top-10) 0.9921 on these held-out queries.
# The new embedder is worth the full rebuild only if it beats that clearly (for example union >= 0.9935).
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -z "${ER_DATASET:-}" ] && { [ -d "$PWD/data/train" ] && ER_DATASET="$PWD/data" || ER_DATASET=/data/dataset; }
export PYTHONUTF8=1 ER_ROOT="${ER_ROOT:-$HOME/er_stage1}" ER_DATASET
# the GPU decides the settings (the instance type is fixed): bf16 only on Ampere+ (A10G, A100, L4); memory decides the micro-batch
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
BF16=$(python -c "import torch; print(1 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 0)")
if [ "${GPU_MB:-0}" -ge 30000 ]; then MB=128; ENC=1024; elif [ "${GPU_MB:-0}" -ge 20000 ]; then MB=64; ENC=512; else MB=16; ENC=256; fi
echo "GPU: ${GPU_NAME:-none} (${GPU_MB:-0} MiB), bf16 $BF16 -> micro-batch $MB, encode batch $ENC"
[ "${GPU_MB:-0}" -ge 14000 ] || { echo "a 568M embedder needs a GPU with at least ~15 GB to fine-tune; set ER_EMBED_BASE=intfloat/multilingual-e5-base for a smaller card"; exit 1; }
export ER_EMBED_BASE="${ER_EMBED_BASE:-BAAI/bge-m3}" ER_EMBED_DIR="${ER_EMBED_DIR:-m3_er}" ER_EMBED_BF16="${ER_EMBED_BF16:-$BF16}"
export ER_TRAIN_BATCH="${ER_TRAIN_BATCH:-512}" ER_MINI_BATCH="${ER_MINI_BATCH:-$MB}" ER_EMBED_LR="${ER_EMBED_LR:-2e-5}" ER_EMBED_EPOCHS="${ER_EMBED_EPOCHS:-1}"
export ER_ENCODE_BATCH="${ER_ENCODE_BATCH:-$ENC}" ER_SEARCH_BATCH="${ER_SEARCH_BATCH:-$ENC}"
mkdir -p "$ER_ROOT/logs"
for f in normalized/source1.parquet normalized/source2.parquet normalized/source3.parquet normalized/eval_queries.parquet normalized/eval_sparse.parquet; do
  [ -f "$ER_ROOT/$f" ] || { echo "missing $ER_ROOT/$f (upload the stage-1 folder)"; exit 1; }
done
echo "$(date +%H:%M:%S) fine-tune $ER_EMBED_BASE -> $ER_ROOT/models/$ER_EMBED_DIR"
[ -f "$ER_ROOT/models/$ER_EMBED_DIR/model.safetensors" ] || python -u embed.py train 2>&1 | tee "$ER_ROOT/logs/stage1_train.log" | grep -E "training pairs|loss|saved|Error|Traceback" 
echo "$(date +%H:%M:%S) held-out dense search + recall"
python -u embed.py eval 2>&1 | tee "$ER_ROOT/logs/stage1_eval.log" | grep -E "sparse|dense|union|Error|Traceback"
echo "reference (current fine-tuned e5-small): dense @10 0.9907, union top10 each 0.9921"
# ---- results for download: $ER_ROOT/stage1_results/ (small) and the fine-tuned model as one archive
OUT="$ER_ROOT/stage1_results"; mkdir -p "$OUT"
cp -f "$ER_ROOT"/logs/stage1_*.log "$OUT/" 2>/dev/null
python - "$ER_ROOT" <<'PY' | tee "$OUT/RESULT.txt"
import re, sys
from pathlib import Path
log = (Path(sys.argv[1]) / "logs" / "stage1_eval.log").read_text(errors="replace")
dense = re.search(r"^dense\s.*@10:([0-9.]+)", log, re.M)
union = re.search(r"union top10 each: recall ([0-9.]+)", log)
d, u = (float(dense.group(1)) if dense else None), (float(union.group(1)) if union else None)
import os
print(f"embedder: {os.environ.get('ER_EMBED_BASE')} fine-tuned on the training pairs (no test data)")
print(f"held-out owner recall: dense top-10 {d}   union (sparse + dense top-10) {u}")
print("current e5-small:      dense top-10 0.9907   union 0.9921")
if u is None:
    print("VERDICT: no recall found in logs/stage1_eval.log - see the log")
else:
    print("VERDICT:", "PASS - worth Stage 2 (union >= 0.9935)" if u >= 0.9935 else ("BORDERLINE - small gain, Stage 2 probably not worth 10-14 h" if u > 0.9921 else "FAIL - not better than the current embedder, stop here"))
PY
echo "$(date +%H:%M:%S) packing the fine-tuned model"
tar czf "$OUT/${ER_EMBED_DIR}_model.tgz" -C "$ER_ROOT/models" --exclude=ckpt "$ER_EMBED_DIR" && ls -la "$OUT"
[ -n "${S3_OUT:-}" ] && aws s3 sync "$OUT" "$S3_OUT" --only-show-errors && echo "copied to $S3_OUT"
echo "results: $OUT  (RESULT.txt = verdict; *_model.tgz = the embedder, needed only for Stage 2)"
echo "$(date +%H:%M:%S) STAGE 1 DONE"
