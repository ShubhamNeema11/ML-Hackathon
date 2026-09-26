#!/usr/bin/env bash
# E2: better cross-encoder (all training records mined, 3 epochs) -> re-score -> features -> ranker B2. Everything tagged _v2 / _ce2.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
export ER_CE_TAG=_v2 ER_CE_MODEL_DIR=ce_er2 ER_STAGE1=ranker_a.txt
step() { echo "== $(date +%H:%M:%S) $*"; }
step "mine (all training records)";   ER_FEAT_TAG=_s ER_CE_SAMPLE=1.0 python -u crossenc.py mine
step "train (3 epochs)";              ER_FEAT_TAG=_s ER_CE_EPOCHS=3 python -u crossenc.py train
step "score train";                   ER_FEAT_TAG=_s python -u crossenc.py score train
step "score eval";                    ER_FEAT_TAG=_s python -u crossenc.py score eval
step "features train";                ER_CE=1 ER_FEAT_TAG=_ce2 python -u ranker.py features train
step "features eval";                 ER_CE=1 ER_FEAT_TAG=_ce2 python -u ranker.py features eval
step "fit ranker B2";                 ER_CE=1 ER_FEAT_TAG=_ce2 ER_MODEL=ranker_b2.txt ER_DECISION=decision_b2.json python -u ranker.py fit
step "E2 DONE"
