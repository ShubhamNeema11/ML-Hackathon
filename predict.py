"""Score the test set and write the submission files.

  python predict.py score                 rank every blocked test pair -> normalized/pred/part*.parquet (all pairs + prob)
  python predict.py write <threshold>     -> output/matching_results.tsv and output/candidate_pairs.tsv

`score` is the expensive step and is done once; `write` is cheap so the threshold can be changed freely.
IDs are mapped to UInt32 indices (normalized/test_ids.parquet) so 180M pairs fit comfortably in memory.

Decision rule: every S2/S3 record has at most one owner in S1 (verified on the training ground truth), so each
record keeps only its single best candidate, and only when the ranker's probability reaches the threshold.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from block import CHUNK, NORM, ROOT
from ranker import FEATURES, MODEL_PATH, TEXT_COLS, merge_channels, retrieval_features, string_features

OUT = Path(__file__).parent / "output"
PRED = NORM / "pred"


def ids_table() -> pl.DataFrame:
    p = NORM / "test_ids.parquet"
    if p.exists():
        return pl.read_parquet(p)
    ids = pl.concat([pl.read_parquet(NORM / f"test_source{i}.parquet", columns=["entity_id"]) for i in (1, 2, 3)]).with_row_index("idx")
    ids.write_parquet(p)
    return ids


def load_dense(ids: pl.DataFrame) -> pl.DataFrame:
    """All dense candidates as (rec_i, s1_i, score, rank) with UInt32 indices, sorted by rec_i.
    ~100M rows: 1.4 GB as integers, >6 GB as strings. Part files are converted a few at a time."""
    files = sorted((NORM / "cand" / "test_dense").glob("*.parquet"))
    id_rec = ids.rename({"entity_id": "rec", "idx": "rec_i"})
    id_s1 = ids.rename({"entity_id": "s1", "idx": "s1_i"})
    out = []
    for i in range(0, len(files), 12):
        d = pl.concat([pl.read_parquet(f) for f in files[i:i + 12]])
        out.append(d.join(id_rec, on="rec").join(id_s1, on="s1")
                    .select("rec_i", "s1_i", pl.col("dense_score").cast(pl.Float32), "dense_rank"))
    return pl.concat(out).sort("rec_i")


def score():
    t0 = time.time()
    ids = ids_table()
    names = ids["entity_id"]
    n_s1 = pl.scan_parquet(NORM / "test_source1.parquet").select(pl.len()).collect().item()
    q_ids = pl.concat([pl.scan_parquet(NORM / f"test_source{i}.parquet").select("entity_id") for i in (2, 3)]).collect()["entity_id"]
    texts = pl.concat([pl.read_parquet(NORM / f"test_source{i}.parquet", columns=TEXT_COLS) for i in (1, 2, 3)])
    dense = load_dense(ids)
    dense_rec = dense["rec_i"]
    model = lgb.Booster(model_file=str(MODEL_PATH))
    PRED.mkdir(exist_ok=True)
    n_parts = -(-len(q_ids) // CHUNK)
    print(f"{len(q_ids):,} test records in {n_parts} chunks; dense rows {dense.height:,} ({time.time() - t0:.0f}s)", flush=True)
    id_rec = ids.rename({"entity_id": "rec", "idx": "rec_i"})
    id_s1 = ids.rename({"entity_id": "s1", "idx": "s1_i"})
    for n in range(n_parts):
        out = PRED / f"part{n:04d}.parquet"
        if out.exists():  # resumable
            continue
        # chunk n = records [n*CHUNK, (n+1)*CHUNK) of S2||S3, which sit at indices n_s1 + position in `ids`
        lo, hi = n_s1 + n * CHUNK, n_s1 + min((n + 1) * CHUNK, len(q_ids))
        a, b = dense_rec.search_sorted(lo), dense_rec.search_sorted(hi)
        de = dense.slice(a, b - a)
        de = de.select(pl.Series("rec", names.gather(de["rec_i"].to_numpy())), pl.Series("s1", names.gather(de["s1_i"].to_numpy())),
                       "dense_score", "dense_rank")
        sp = pl.read_parquet(NORM / "cand" / "test_sparse" / f"part{n:04d}.parquet")
        c = retrieval_features(merge_channels(sp, de))
        need = pl.concat([c.select(pl.col("rec").alias("entity_id")), c.select(pl.col("s1").alias("entity_id"))]).unique()
        f = string_features(c, texts.join(need, on="entity_id", how="semi"))
        p = model.predict(f.select(FEATURES).cast(pl.Float32).to_numpy())
        f = (f.select("rec", "s1").with_columns(pl.Series("p", p, dtype=pl.Float32))
              .join(id_rec, on="rec").join(id_s1, on="s1").select("rec_i", "s1_i", "p"))
        f.write_parquet(out)
        if n % 10 == 0:
            print(f"  chunk {n + 1}/{n_parts}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"scored ({time.time() - t0:.0f}s)", flush=True)


def _write_tsv(df: pl.DataFrame, f):
    f.write(df.write_csv(separator="\t", include_header=False, quote_style="never").encode("utf-8"))


def write(threshold: float):
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    ids = ids_table()
    names = ids["entity_id"]
    n_s1 = pl.read_parquet(NORM / "test_source1.parquet", columns=["entity_id"]).height  # S1 rows come first in ids
    parts = sorted(PRED.glob("part*.parquet"))
    best, pairs = [], []
    for p in parts:
        d = pl.read_parquet(p)
        pairs.append(d.select("s1_i", "rec_i"))
        best.append(d.sort("p", descending=True).group_by("rec_i", maintain_order=True).head(1))
    best = pl.concat(best).filter(pl.col("p") >= threshold)
    print(f"threshold {threshold}: {best.height:,} records assigned an S1 owner ({time.time() - t0:.0f}s)", flush=True)

    def lists(pr: pl.DataFrame) -> pl.DataFrame:
        """S1 index -> comma-separated S2/S3 ids, one row for every S1 entity."""
        g = (pr.with_columns(pl.Series("rec", names.gather(pr["rec_i"].to_numpy())))
               .group_by("s1_i").agg(pl.col("rec").sort().str.join(",").alias("ids")))
        allS1 = pl.DataFrame({"s1_i": np.arange(n_s1, dtype=np.uint32)})
        return allS1.join(g, on="s1_i", how="left").with_columns(pl.col("ids").fill_null("")).sort("s1_i")

    m = lists(best.select("s1_i", "rec_i"))
    m = m.with_columns(pl.Series("source1_entity_id", names.gather(m["s1_i"].to_numpy()))).select("source1_entity_id", "ids")
    with open(OUT / "matching_results.tsv", "wb") as f:
        f.write(b"source1_entity_id\tmatched_entity_ids\n")
        _write_tsv(m, f)
    print(f"matching_results.tsv: {m.height:,} S1 rows, {(m['ids'] == '').sum():,} empty ({time.time() - t0:.0f}s)", flush=True)

    allp = pl.concat(pairs).sort("s1_i")
    del pairs
    with open(OUT / "candidate_pairs.tsv", "wb") as f:
        f.write(b"source1_entity_id\tcandidate_entity_ids\n")
        step = 200_000
        for lo in range(0, n_s1, step):
            hi = min(lo + step, n_s1)
            sl = allp.filter((pl.col("s1_i") >= lo) & (pl.col("s1_i") < hi))
            c = lists(sl).filter((pl.col("s1_i") >= lo) & (pl.col("s1_i") < hi))
            c = c.with_columns(pl.Series("source1_entity_id", names.gather(c["s1_i"].to_numpy()))).select("source1_entity_id", "ids")
            _write_tsv(c, f)
    print(f"candidate_pairs.tsv written ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score()
    elif sys.argv[1] == "write":
        write(float(sys.argv[2]))
