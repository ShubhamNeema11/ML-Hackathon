#!/usr/bin/env bash
# Full run on an AWS GPU instance that RETRAINS the models that were only trained on small samples on the laptop and reuses the rest: the weights
# of the retrained components (default: cross-encoder, LightGBM rankers, no-address specialist) are trained here, then evaluated on the held-out entities,
# then the real test set is scored and the final matching written. The fine-tuned e5 embedder and all data / candidate files of the laptop run are reused.
# Prerequisite: the laptop run's normalized/ and models/ in $ER_ROOT (bash upload_to_s3.sh copies them), the dataset in $ER_DATASET (train/, test/,
# ../utils/validate_submission.py), `bash aws/setup.sh` done. Run from the repository root:
#
#   tmux new -s er
#   AUTO_STOP=1 S3_OUT=s3://your-bucket/er_full bash aws/run_full.sh
#
# What runs (python main.py --retrain ..., every step its own process, logs in $ER_ROOT/logs/<step>.log, timings in logs/timeline.tsv):
#   normalize -> sparse + dense blocking (e5 fine-tuned here) -> no-address blocking (TF-IDF) + features -> ranker A -> cross-encoder (address pairs only)
#   -> ranker B -> choose A/B -> no-address specialist -> joint decision rule -> score test -> patch no-address rows -> write -> validate -> report
# and always ends with $ER_ROOT/deliverables/: matching_results.tsv, candidate_pairs.tsv, RUN_REPORT.md, logs.tgz, models_manifest.tsv.
#
# RETRAIN (default ce,rankers,noaddr; also embed, or all) decides which weights are trained again. `main.py --retrain` moves those models and everything
# derived from them (features, cross-encoder scores, score parts, old output) to $ER_ROOT/_replaced/<time>/, so their steps run again; a dependent component is
# added automatically (rankers -> ce; embed -> all). Re-running the same command resumes THIS run (finished steps are skipped). Environment options:
#   ER_ROOT=/data/er_work        work folder with the laptop run's normalized/ + models/      ER_DATASET=/data/dataset
#   RETRAIN=ce,rankers,noaddr    components whose weights are trained again (embed = e5 embedder: only if you decide to change it)
#   AUTO_STOP=1                  report + upload + shut the instance down (2 min after success, 30 min after a failure; `sudo shutdown -c` cancels)
#   S3_OUT=s3://...              copy deliverables/ there at the end
#   ER_EMBED_PAIRS=600000        embedder training pairs, only used when RETRAIN contains embed (2000000 = every S1 entity)
#   ER_CE_SAMPLE=1.0             share of the training records mined for cross-encoder pairs (default 0.33; 1.0 = all, several times more training pairs)
#   TRAIN_FRAC=0.10  EVAL_FRAC=0.05   share of records for ranker training / held-out evaluation queries (as on the laptop)
#   ER_CE_BASE=...               cross-encoder base model (default multilingual-e5-small; e.g. BAAI/bge-reranker-v2-m3 with ER_CE_DTYPE=bf16 ER_CE_LR=2e-5)
#   SKIP_CE=1                    no cross-encoder (ranker A only)          DRYRUN=1   print every command, run nothing
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONUTF8=1 ER_ROOT="${ER_ROOT:-/data/er_work}" ER_DATASET="${ER_DATASET:-/data/dataset}"
export ER_CE_SAMPLE="${ER_CE_SAMPLE:-1.0}"
RETRAIN="${RETRAIN:-ce,rankers,noaddr}"
AUTO_STOP="${AUTO_STOP:-0}"; DRYRUN="${DRYRUN:-0}"
ARGS=(--retrain "$RETRAIN" --dataset "$ER_DATASET" --root "$ER_ROOT" --train-frac "${TRAIN_FRAC:-0.10}" --eval-frac "${EVAL_FRAC:-0.05}")
[ "${SKIP_CE:-0}" = 1 ] && ARGS+=(--skip-ce)
mkdir -p "$ER_ROOT"

if [ "$DRYRUN" = 1 ]; then
  echo "[dry] ER_ROOT=$ER_ROOT ER_DATASET=$ER_DATASET RETRAIN=$RETRAIN ER_CE_SAMPLE=$ER_CE_SAMPLE  (a dry run moves nothing)"
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
