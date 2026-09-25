# Business Entity Resolution

Given three noisy business sources, find for every Source 1 (S1) entity the matching Source 2 / Source 3 records.
S1 is deduplicated, so every S2/S3 record has at most one owner in S1 (verified on the training ground truth:
7,638,365 matched pairs, 7,638,365 distinct S2/S3 ids). The pipeline therefore works record -> S1 and lets each
record keep only its single best candidate.

```
normalize.py  ->  block.py (sparse) + embed.py (dense)  ->  ranker.py  ->  predict.py
   clean text        candidate generation, top-10 each       LightGBM       score test, write TSVs
```

## Environment

Python 3.11, `pip install -r requirements.txt` (install the CUDA build of torch first, see the file header).
A GPU with 6 GB is enough for the dense channel. Peak RAM is about 9 GB (blocking) so 16 GB machines work.

Paths default to the layout of the challenge bundle; override with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `ER_DATASET` | folder containing `train/` and `test/` | the local `student_resource/dataset` |
| `ER_ROOT` | folder that holds `normalized/` and `models/` | this folder |
| `ER_WORKERS` | processes used by normalization | min(8, CPUs) |
| `ER_CHUNK` | queries per search chunk (also part-file size) | 50000 |

## Run everything with one command

```bash
python main.py --dataset /path/to/dataset --root /path/to/workdir     # resumable; --list, --dry-run, --skip-ce, --from/--to/--only
```
`main.py` runs every step below in order (as separate processes, logs in `logs/`), sets thread counts and batch sizes from the
machine, chooses between ranker A and B (with cross-encoder features) on held-out data, writes both output files and runs the
organisers' validator. For AWS see [aws/README_AWS.md](aws/README_AWS.md) and `aws/setup.sh`.

## Reproduce end to end, step by step

```bash
# 1. normalize train and test (mojibake, Indic scripts, US / India / France addresses, legal forms)
python normalize.py train
python normalize.py test

# 2. held-out evaluation queries + sparse candidates (10% of S1 entities are never used for training)
python block.py eval 0.05

# 3. fine-tune multilingual-e5-small on (S2/S3 record, S1 owner) pairs, then evaluate both channels
python embed.py train
python embed.py eval

# 4. candidates for ranker training (10% of records) and for the test set
python block.py train 0.10
python embed.py search train
python block.py test
python embed.py search test

# 5. ranker: features, LightGBM fit with early stopping on the held-out entities
python ranker.py features train
python ranker.py features eval
python ranker.py fit

# 5b. (optional) name-only extras for records WITHOUT an address: +5 candidates each (each stage its own process)
python extras.py sparse train && python extras.py dense train && python extras.py build train
python extras.py build eval
python extras.py sparse test  && python extras.py dense test  && python extras.py build test

# 5c. features now include structured address numbers, legal-form / name-word differences and states (structfeat.py);
#     `fit` mines hard positives / hard negatives, weights them, and tunes the decision rule (models/decision.json)
python ranker.py features train && python ranker.py features eval
python ranker.py fit

# 5d. (optional) cross-encoder on the uncertain band; its score becomes a LightGBM feature (ER_CE=1)
python crossenc.py mine && python crossenc.py train
python crossenc.py score train && python crossenc.py score eval          # then: ER_CE=1 python ranker.py features ... / fit
python crossenc.py score test                                             # after `predict.py score` (needs stage-1 test scores)

# 6. score every blocked test pair, then write the submission files (no threshold argument = tuned decision rule)
python predict.py score
python predict.py write            # or: python predict.py write 0.80  (single threshold)
```

Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Validate with the organisers' helper:
`python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Method summary

1. **Normalization** (`normalize.py`): fixes mojibake (ftfy), NFKC + casefold, strips accents on Latin letters only,
   canonicalises legal forms (Pvt/Private/`प्राइवेट` -> `pvt`, SARL, SAS, LLC...), expands street abbreviations, parses
   state / city / postal code per country (US, India, France; unknown countries get generic handling), and keeps both the
   original-script text and a Latin transliteration.
2. **Blocking, two channels, top-10 each, union**:
   * sparse: IDF-weighted cosine over hashed keys (name tokens, prefixes/suffixes, consonant skeleton, 4-grams, initials,
     address words, house numbers and combined keys), country-scoped;
   * dense: `intfloat/multilingual-e5-small` (MIT, 118M) fine-tuned with in-batch negatives, exact top-10 cosine on GPU.
   Held-out recall of the true owner: sparse@10 94.4%, dense@10 99.1%, union 99.2% at ~18 candidates per record.
3. **Ranker** (`ranker.py`): LightGBM on 28 country-agnostic features (retrieval scores/ranks/margins, five fuzzy name
   similarities, address similarity, city/state/postal/house-number agreement, legal form, missing-field flags).
4. **Decision** (`predict.py`): each record keeps its best candidate if the probability reaches the threshold (0.80,
   chosen on held-out entities with the orphan share reweighted to the real 26%).

No external data, APIs or look-ups are used; the only pretrained model is multilingual-e5-small (MIT).
