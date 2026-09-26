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

## Upgrade run: big cross-encoder + pass-2 (`aws/run_pipeline.sh`, unattended and documented)
Prerequisite: the laptop run's `normalized/` and `models/` in `$ER_ROOT` (default `/data/er_work`; `bash upload_to_s3.sh` uploads only what is needed, about 12-13 GB),
the dataset in `/data/dataset`, `bash aws/setup.sh` done. Then, in tmux:

    AUTO_STOP=1 S3_OUT=s3://<bucket>/er_results bash aws/run_pipeline.sh all      # add LEAN=1 to skip the big cross-encoder

`all` runs pilot -> big cross-encoder -> mirror -> pass 2 -> real test set, with three automatic gates (a step only feeds the next if it earned it):
pilot vs the small cross-encoder, B4 vs B2 (> 0.0005 on the half no threshold was tuned on), pass 2 vs pass 1 (> 0.0005, same procedure).
It resumes where it stopped if you run the same command again, and ends with `$ER_ROOT/deliverables/`:
`matching_results.tsv`, `candidate_pairs.tsv`, `RUN_REPORT.md` (choices, held-out numbers, per-step timeline, cost estimate, sanity checks, environment), `logs.tgz`, `models_manifest.tsv`.
`AUTO_STOP=1` copies that to `S3_OUT` and stops the instance (2 min after success, 30 min after a failure; `sudo shutdown -c` cancels), so no idle billing.
`DRYRUN=1 bash aws/run_pipeline.sh all` prints every command without running anything.

Estimated time on one g5.2xlarge (A10G 24 GB, 8 vCPU), from laptop timings (not measured on AWS):

| phase | what | time |
|---|---|---|
| `pilot` | round-3 hard pairs, big cross-encoder for 1,500 steps, score the held-out band, gate | 0.5 h |
| `bigce` | full training (2 epochs), score train/eval bands, B4, gate. The CPU-only mirror blocking runs alongside | 2 h |
| `mirror` | test-time pipeline over the training records (remainder after the overlap) | 3 h |
| `pass2` | fit pass 2, gate | 0.5 h |
| `finish` | real test set: scores, pass 2, files, validator | 2.75 h |
| **full run** | | **about 9 h, about $16** (range $11-22) |
| **LEAN=1** (no big cross-encoder) | mirror 3.2 h + pass 2 0.5 h + finish 1.4 h | **about 5 h, about $9** |
| pilot NO-GO | pilot + lean | about 5.5 h, about $10 |

Cost assumes about $1.6/h on-demand in ap-southeast-2 plus about $1.5 of storage (the price could not be queried; spot is $0.84-0.91/h but the spot quota is 0).
Budget $30 for reruns. Pieces that are CPU-bound (blocking, LightGBM) run slower on 8 vCPUs than on the laptop's 16 threads; if the 32-vCPU quota is approved, use a g5.8xlarge for the CPU stages.

## The full run: `aws/run_full.sh` (SageMaker JupyterLab or EC2) — use this one

One command trains what has to be trained, evaluates on the held-out entities, scores the test set and writes the final matching:
address records -> ranker A -> cross-encoder -> ranker B -> name + address rerankers (bge-reranker-v2-m3, listwise loss on hard negatives) -> ranker C
(A/B/C chosen on held-out data); no-address records -> TF-IDF blocking + own LightGBM on all ~310k no-address training records (no cross-encoder);
then pass 2 (mirror over the training records, used only if it wins by > 0.0005), final files, report, validator. The e5 embedder is reused, not retrained.

    git pull && bash aws/sagemaker_setup.sh            # must print READY
    nohup bash aws/run_full.sh > run_full.out 2>&1 &    # tail -f run_full.out ; DRYRUN=1 prints the plan

`RETRAIN` (default `ce,rankers,noaddr,rerank,pass2`) decides which weights are trained again; `main.py --retrain` moves those models and everything computed
from them to `_replaced/<time>/` first, so an uploaded laptop `normalized/` + `models/` can never make a retrained step look done. `RERANK=0` / `PASS2=0` leave those parts out.
Results: `$ER_ROOT/deliverables/` (matching_results.tsv, candidate_pairs.tsv, RUN_REPORT.md, logs.tgz, models_manifest.tsv).
`aws/run_pipeline.sh` (below) is the older single-pipeline upgrade script; it does not know the two pipelines.

## Rules
Only code and the provided data are used. No AWS AI service (Bedrock, Comprehend, Entity Resolution, ...) and no external data
or API is involved, in line with the competition's fair-play rules. The pretrained model is multilingual-e5-small (MIT licence).
