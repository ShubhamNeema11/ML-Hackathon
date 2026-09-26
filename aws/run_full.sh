#!/usr/bin/env bash
# The full run on a GPU machine (SageMaker JupyterLab or EC2): trains what has to be trained, evaluates on the held-out entities, scores the real
# test set and writes the final matching. One command; every step is its own process with its own log; a failed step stops the run and the same
# command resumes it (finished steps are skipped).
#
# SageMaker JupyterLab (default paths): repo ~/ML-Hackathon, dataset ~/ML-Hackathon/data/{train,test}, work folder ~/er_work.
#   git pull && bash aws/sagemaker_setup.sh          # once: checks GPU / disk / dataset, unpacks e5_er.zip, builds .venv; must print READY
#   nohup bash aws/run_full.sh > run_full.out 2>&1 &  # then:  tail -f run_full.out
# EC2: bash aws/setup.sh, ER_ROOT=/data/er_work ER_DATASET=/data/dataset, tmux, AUTO_STOP=1 S3_OUT=s3://your-bucket/er_full.
#
# WHAT RUNS (python main.py --retrain $RETRAIN ...; see the docstring of main.py for the details):
#   records WITH an address:   ranker A -> cross-encoder (address pairs) -> ranker B -> name + address rerankers (pretrained bge-reranker-v2-m3, listwise /
#                              contrastive loss on hard negatives) -> ranker C; A, B or C is chosen on the held-out half no threshold was tuned on
#   records WITHOUT an address: char 3-gram TF-IDF blocking -> own LightGBM trained on ALL ~310k no-address training records, no cross-encoder / reranker
#   then: joint decision rule -> test scores -> patch the no-address rows -> pass 2 (mirror over the training records, sibling-support model, used only if it
#   beats pass 1 by > 0.0005) -> final files -> report + deliverables/ -> validator.
# The e5 embedder is NOT retrained (models/e5_er is reused); everything else that depends on retrained weights is recomputed.
#
# WHAT IS REUSED / RETRAINED. RETRAIN (default ce,rankers,noaddr,rerank,pass2; also embed, or all) names the weights that are trained in this run. `main.py --retrain`
# moves those models AND everything computed from them (features, cross-encoder scores, score parts, old output, the pass-2 mirror) to $ER_ROOT/_replaced/<time>/,
# so their steps run again; a dependent component is added automatically. Normalized data and blocking / dense candidate files in $ER_ROOT (from the laptop
# run or from an earlier part of this run) are reused; without them everything is computed. Re-running the command resumes THIS run (a marker file).
#
# Environment options:
#   ER_ROOT=$HOME/er_work        work folder (models/e5_er inside; optionally the laptop run's normalized/)      ER_DATASET=$HOME/ML-Hackathon/data
#   RETRAIN=ce,rankers,noaddr,rerank,pass2    components trained in this run (embed = the e5 embedder: only if you decide to change it)
#   RERANK=0                     no name / address rerankers (saves about 4 h)          PASS2=0     no pass 2 (saves about 4-5 h)      SKIP_CE=1   ranker A only
#   RERANK_BASE=BAAI/bge-reranker-v2-m3   RERANK_SAMPLE=0.1 (share of the training records mined for reranker groups, ~65k groups)   RERANK_EPOCHS=1
#   CE_EPOCHS=3                  epochs of the joint cross-encoder (the laptop's round 2 used 3)
#   ER_CE_SAMPLE=1.0             share of the training records mined for the joint cross-encoder (default here 1.0; the laptop run used 0.33)
#   ER_CHUNK=50000               records per blocking chunk; MUST equal the value used for reused test blocking (the laptop run used 50000)
#   TRAIN_FRAC=0.10  EVAL_FRAC=0.05   share of records for ranker training / held-out evaluation queries (as on the laptop)
#   KEEP_MIRROR=1                keep the pass-2 mirror's candidate / score folders (about 20 GB) after pass 2
#   AUTO_STOP=1  S3_OUT=s3://... EC2 only: copy deliverables/ to S3 and shut the instance down            DRYRUN=1   print every step and command, run nothing
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -z "${ER_DATASET:-}" ] && { [ -d "$PWD/data/train" ] && ER_DATASET="$PWD/data" || ER_DATASET=/data/dataset; }
export PYTHONUTF8=1 ER_ROOT="${ER_ROOT:-$HOME/er_work}" ER_DATASET
export ER_CHUNK="${ER_CHUNK:-50000}" ER_CE_SAMPLE="${ER_CE_SAMPLE:-1.0}"
RETRAIN="${RETRAIN:-ce,rankers,noaddr,rerank,pass2}"
AUTO_STOP="${AUTO_STOP:-0}"; DRYRUN="${DRYRUN:-0}"
ARGS=(--retrain "$RETRAIN" --dataset "$ER_DATASET" --root "$ER_ROOT" --train-frac "${TRAIN_FRAC:-0.10}" --eval-frac "${EVAL_FRAC:-0.05}")
[ "${SKIP_CE:-0}" = 1 ] && ARGS+=(--skip-ce)
[ "${RERANK:-1}" = 0 ] && ARGS+=(--no-rerank)
[ "${PASS2:-1}" = 0 ] && ARGS+=(--no-pass2)
mkdir -p "$ER_ROOT"

if [ "$DRYRUN" = 1 ]; then
  echo "[dry] ER_ROOT=$ER_ROOT ER_DATASET=$ER_DATASET RETRAIN=$RETRAIN ER_CHUNK=$ER_CHUNK ER_CE_SAMPLE=$ER_CE_SAMPLE RERANK=${RERANK:-1} PASS2=${PASS2:-1}  (a dry run moves nothing and checks nothing)"
  python main.py "${ARGS[@]}" --dry-run
  exit 0
fi

finish() {
  local rc=$?
  trap - EXIT
  ER_ROOT="$ER_ROOT" STATUS="$([ $rc -eq 0 ] && echo SUCCESS || echo "FAILED (exit code $rc): see logs/timeline.tsv and the last step log")" python aws/report_full.py || echo "report failed"
  [ -n "${S3_OUT:-}" ] && aws s3 sync "$ER_ROOT/deliverables" "$S3_OUT" --only-show-errors && echo "deliverables copied to $S3_OUT"
  if [ "$AUTO_STOP" = 1 ]; then
    local m=2; [ $rc -ne 0 ] && m=30
    echo "instance stops in $m minutes (cancel with: sudo shutdown -c)"; sudo shutdown -h +$m
  fi
  exit $rc
}
trap finish EXIT

python main.py "${ARGS[@]}"
