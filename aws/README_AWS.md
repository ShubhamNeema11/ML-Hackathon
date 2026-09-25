# Running the pipeline on AWS

The whole pipeline is `python main.py`. This page covers only what is specific to AWS.

## Instance
| | vCPU | RAM | GPU | approx. price (on-demand) |
|---|---|---|---|---|
| **g5.8xlarge (recommended)** | 32 | 128 GB | A10G, 24 GB | about $2.4 / h |
| g5.4xlarge (cheaper, slower) | 16 | 64 GB | A10G, 24 GB | about $1.6 / h |

* Ubuntu with NVIDIA drivers (the *Deep Learning AMI (Ubuntu)* has them). 300 GB gp3 disk (a full run writes about 60 GB).
* `main.py` reads the machine (CPUs, RAM, GPU memory) and sets thread counts, chunk sizes and GPU batch sizes itself
  (`ER_WORKERS`, `ER_THREADS`, `ER_CHUNK`, `ER_SEARCH_BATCH`, ...). Set any of them yourself to override.
* Use the same instance for the whole run: `ER_CHUNK` (records per chunk) must not change between `block_test` and `score_a`;
  `predict.py score` stops with a clear message if it did.

## Steps
1. Launch the instance, copy the repo onto it (`git clone` or `scp`), and upload the challenge data
   (`dataset/train/*.tsv`, `dataset/test/*.tsv`, and `utils/validate_submission.py` next to `dataset/`) to S3 or with `scp`.
2. `bash aws/setup.sh` (installs Python 3.11 if the image only has 3.10; optionally `S3_DATASET=s3://bucket/path/dataset bash aws/setup.sh`). It creates a virtualenv,
   installs the pinned requirements and the CUDA build of torch, and copies the data to `/data/dataset`.
3. `tmux new -s er`, then
   `source .venv/bin/activate && python main.py --dataset /data/dataset --root /data/er_work`
4. When it finishes, `/data/er_work/output/` holds `matching_results.tsv` and `candidate_pairs.tsv`, already validated.
   Copy them off before stopping the instance: `aws s3 cp /data/er_work/output s3://your-bucket/output --recursive`
   (or `scp`). Trained models are in `/data/er_work/models/`.
5. **Stop the instance** when done. Anything not on S3 or your own disk disappears with an instance you terminate.

## Useful options
* `python main.py --list` shows every step and whether it is done; `--dry-run` prints the commands.
* `--skip-ce` leaves out the cross-encoder (shorter). Without it `main.py` trains ranker A and ranker B (A plus cross-encoder
  features) and keeps whichever scores better on the held-out half that no threshold was tuned on (B has to win by more than 0.0005).
* `--from <step>` / `--to <step>` / `--only a,b` run part of the pipeline; `--force` repeats finished steps.
* Logs: `/data/er_work/logs/<step>.log`.

## Rough time (estimates from the runs on a laptop with an RTX 4050, not measured on AWS)
Normalization ~5 min - blocking (sparse + dense, all three query sets) ~1-1.5 h - e5 fine-tuning ~10-15 min - rankers A and B ~15-30 min
in total - cross-encoder ~30-45 min - scoring the test set once or twice ~30-60 min. Plan for **4 to 6 hours** on a g5.8xlarge.

## Rules
Only code and the provided data are used. No AWS AI service (Bedrock, Comprehend, Entity Resolution, ...) and no external data
or API is involved, in line with the competition's fair-play rules. The pretrained model is multilingual-e5-small (MIT licence).
