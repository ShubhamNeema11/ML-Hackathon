#!/usr/bin/env bash
# Raw-name reranker for the NAME-ONLY (no-address) pipeline + raw-name features in the specialist. Training data only; resumable; logs/nar_*.log.
cd "$(dirname "$0")"
export PYTHONUTF8=1 ER_NA_TFIDF=1 ER_NA_RAW=1 ER_CODE_DROPOUT=0.5 ER_CODE_UNSEEN=1 ER_FR_LEGAL=0 ER_FR_ADDR=0 ER_FR_NAME=0 ER_FR_GENERIC=0
RR=(env ER_CE_TAG=_nar ER_CE_TEXT=raw_name ER_CE_MODEL_DIR=ce_nar ER_CE_BASE=intfloat/multilingual-e5-small ER_CE_MAXLEN=48 ER_CE_DTYPE=fp16 ER_CE_SCORE_BATCH=512)
step() { local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date +%H:%M:%S) [$name] done already"; return 0; fi
  echo "$(date +%H:%M:%S) [$name] start"; local t=$(date +%s)
  if ! "$@" >> "logs/nar_$name.log" 2>&1; then echo "[$name] FAILED:"; tail -n 15 "logs/nar_$name.log"; exit 1; fi
  echo "$(date +%H:%M:%S) [$name] done ($(( ($(date +%s) - t) / 60 )) min)"; }
step train "models/ce_nar/model.safetensors" "${RR[@]}" ER_CE_LOSS=listwise ER_CE_GROUP=7 ER_CE_BATCH=56 ER_CE_ACCUM=1 ER_CE_EPOCHS=1 ER_CE_LR=3e-5 python -u crossenc.py train
echo "RR TRAIN DONE $(date +%H:%M:%S)"
