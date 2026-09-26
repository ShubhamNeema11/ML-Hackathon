#!/usr/bin/env bash
# E3 = cross-encoder round 3: warm start from round 2, trained on the round-2 pairs PLUS the pairs the stronger ranker B2 still finds hard.
# Everything is tagged _v3 / _ce3; each step is skipped when its output already exists (resumable).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
PY="${ER_PYTHON:-python}"   # the exact interpreter (the caller passes it: PATH may not contain python)
export ER_CE_TAG=_v3 ER_CE_MODEL_DIR=ce_er3
step() { echo "== $(date +%H:%M:%S) $*"; }
[ -f normalized/cepairs_v3.parquet ] || { step "mine (hard for ranker B2)";  ER_FEAT_TAG=_ce2 ER_STAGE1=ranker_b2.txt ER_CE_SAMPLE=1.0 "$PY" -u crossenc.py mine; }
[ -f models/ce_er3/model.safetensors ] || { step "train (warm start from round 2, 3 more epochs)"; ER_FEAT_TAG=_s ER_CE_EXTRA_PAIRS=_v2 ER_CE_INIT=ce_er2 ER_CE_EPOCHS=3 ER_CE_LR=2e-5 "$PY" -u crossenc.py train; }
[ -f normalized/ce_train_v3.parquet ] || { step "score train"; ER_FEAT_TAG=_s ER_STAGE1=ranker_a.txt "$PY" -u crossenc.py score train; }
[ -f normalized/ce_eval_v3.parquet ]  || { step "score eval";  ER_FEAT_TAG=_s ER_STAGE1=ranker_a.txt "$PY" -u crossenc.py score eval; }
[ -f normalized/feat_eval_ce3.parquet ] || { step "features train"; ER_CE=1 ER_FEAT_TAG=_ce3 "$PY" -u ranker.py features train
                                             step "features eval";  ER_CE=1 ER_FEAT_TAG=_ce3 "$PY" -u ranker.py features eval; }
[ -f models/decision_b3.json ] || { step "fit ranker B3"; ER_CE=1 ER_FEAT_TAG=_ce3 ER_MODEL=ranker_b3.txt ER_DECISION=decision_b3.json "$PY" -u ranker.py fit; }
step "E3 DONE"
