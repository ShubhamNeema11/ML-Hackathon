#!/usr/bin/env bash
# STAGE 2, AWS GPU part (only if Stage 1 passed): inference with the two models Stage 1 trained, nothing else. No test data is learned from.
#   - dense candidates with the new embedder (models/m3_er): ranker-training queries, all no-address training records (specialist), whole test set
#   - the new reranker (models/rr_er) on the uncertain bands the current cross-encoder scores (the same pairs: ce_train_v2 / ce_test_v2)
# Everything is tagged _m3 / _rr, packed into ~/er_stage1/stage2_gpu_results.tgz for the laptop (features, rankers, scoring stay there).
#
#   cd ~/ML-Hackathon && git pull && source .venv/bin/activate
#   aws s3 sync s3://<bucket>/er_stage2 ~/er_stage1 --only-show-errors      (the files of stage2_upload.py, into the SAME folder as Stage 1)
#   STAGE2_CHECK=1 bash aws/stage2_gpu.sh        then:   nohup bash aws/stage2_gpu.sh > stage2.out 2>&1 &
#   (S3_OUT=s3://<bucket>/stage2_results copies the archive to S3)
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -z "${ER_DATASET:-}" ] && { [ -d "$PWD/data/train" ] && ER_DATASET="$PWD/data" || ER_DATASET=/data/dataset; }
export PYTHONUTF8=1 ER_ROOT="${ER_ROOT:-$HOME/er_stage1}" ER_DATASET
export ER_EMBED_BASE="${ER_EMBED_BASE:-BAAI/bge-m3}" ER_EMBED_DIR="${ER_EMBED_DIR:-m3_er}" ER_DENSE_TAG=_m3
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
BF16=$(python -c "import torch; print(1 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 0)")
ENC=$([ "${GPU_MB:-0}" -ge 20000 ] && echo 512 || echo 256)
export ER_ENCODE_BATCH="${ER_ENCODE_BATCH:-$ENC}" ER_SEARCH_BATCH="${ER_SEARCH_BATCH:-$ENC}"
N="$ER_ROOT/normalized"; mkdir -p "$ER_ROOT/logs" "$N/cand"
need=(normalized/source1.parquet normalized/source2.parquet normalized/source3.parquet normalized/test_source1.parquet normalized/test_source2.parquet
      normalized/test_source3.parquet normalized/train_queries.parquet normalized/trainall_queries.parquet normalized/ce_train_v2.parquet
      normalized/ce_test_v2.parquet models/m3_er/model.safetensors models/rr_er/model.safetensors)
for f in "${need[@]}"; do [ -e "$ER_ROOT/$f" ] || { echo "missing $ER_ROOT/$f"; exit 1; }; done
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "torch cannot see the GPU"; exit 1; }
[ "${STAGE2_CHECK:-0}" = 1 ] && { echo "STAGE2 CHECK OK: all inputs found, GPU visible (${GPU_MB} MiB)"; exit 0; }
step() { local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "$ER_ROOT/logs/stage2_$name.log" 2>&1; then echo "[$name] FAILED:"; tail -n 15 "$ER_ROOT/logs/stage2_$name.log"; exit 1; fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"; }
# the held-out dense file of Stage 1 was written without a tag: keep it under the new tag
[ -f "$N/eval_dense_m3.parquet" ] || cp -p "$N/eval_dense.parquet" "$N/eval_dense_m3.parquet"
step dense_train    "$N/cand/train_dense_m3/_DONE"    python -u embed.py search train
step dense_trainall "$N/cand/trainall_dense_m3/_DONE" python -u noaddr.py regular_dense
step dense_test     "$N/cand/test_dense_m3/_DONE"     python -u embed.py search test
RR=(env ER_CE_BASE=BAAI/bge-reranker-v2-m3 ER_CE_MODEL_DIR=rr_er ER_CE_TEXT=joint ER_CE_MAXLEN=128 ER_CE_DTYPE=$([ "$BF16" = 1 ] && echo bf16 || echo fp16)
    ER_CE_SCORE_BATCH=$([ "${GPU_MB:-0}" -ge 20000 ] && echo 512 || echo 256) ER_CE_TAG=_rr)
step rr_train_band  "$N/ce_train_rr.parquet"          "${RR[@]}" ER_CE_PAIRS="$N/ce_train_v2.parquet" python -u crossenc.py score train
step rr_test_band   "$N/ce_test_rr.parquet"           "${RR[@]}" ER_CE_PAIRS="$N/ce_test_v2.parquet" python -u crossenc.py score test
[ -f "$N/ce_eval_rr.parquet" ] || cp -p "$N/ce_eval_rrv.parquet" "$N/ce_eval_rr.parquet"
echo "$(date +%H:%M:%S) packing"
tar czf "$ER_ROOT/stage2_gpu_results.tgz" -C "$ER_ROOT" normalized/eval_dense_m3.parquet normalized/cand/train_dense_m3 normalized/cand/trainall_dense_m3 \
    normalized/cand/test_dense_m3 normalized/ce_train_rr.parquet normalized/ce_eval_rr.parquet normalized/ce_test_rr.parquet
ls -la "$ER_ROOT/stage2_gpu_results.tgz"
[ -n "${S3_OUT:-}" ] && aws s3 cp "$ER_ROOT/stage2_gpu_results.tgz" "$S3_OUT/stage2_gpu_results.tgz" --only-show-errors && echo "copied to $S3_OUT"
echo "$(date +%H:%M:%S) STAGE 2 GPU DONE: bring stage2_gpu_results.tgz to the laptop (unpack into the repository folder) and run run_stage2_local.sh"
