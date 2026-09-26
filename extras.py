"""Name-only extra candidates for records WITHOUT an address.

A record with no address cannot be separated from the many S1 entities that share its (often generic) name, so the
address-aware channels rank its true owner deep. For exactly those records we run a name-only sparse search and a
name-only dense search, fuse the two rankings (reciprocal-rank fusion) and add the best N_EXTRA candidates that the
regular top-10 sparse U top-10 dense set does not already contain.
Held-out: +5 such extras lift recall 99.21% -> 99.35% while adding only 0.1 candidates per record on average.

Every stage is its own process (polars-heavy sparse work and torch GPU work segfault when mixed in one process here):

  python extras.py sparse  train|test     name-only sparse search for the no-address records
  python extras.py dense   train|test     name-only dense search (GPU)
  python extras.py sparse|dense eval   name-only searches for the held-out no-address queries
  python extras.py build   train|eval|test  fuse -> normalized/cand/<split>_extra.parquet  (rec, s1, extra_rank 1..5)

`eval` = the held-out queries of `block.py eval` (falls back to the files of ablate_extra.py if they exist and the new ones do not).
`train` = the ranker-training queries from `block.py train`.
"""
import os
import sys
import time

import polars as pl

from block import NORM, SparseIndex, load

N_EXTRA = int(os.environ.get("ER_EXTRA_N", 5))   # ER_EXTRA_N / ER_EXTRA_TAG: deeper lists in a separate file (see noaddr.py)
EXTRA_TAG = os.environ.get("ER_EXTRA_TAG", "")
DEPTH = 30
RRF = 60.0


def src_of(split: str) -> tuple[str, str]:
    """(source split for load(), file prefix) - held-out eval and ranker-train queries both come from the train files."""
    return ("test", "test_") if split == "test" else ("train", "")


def no_addr_ids(split: str) -> pl.DataFrame:
    """entity_id of the S2/S3 records without an address that belong to this split's query set."""
    _, prefix = src_of(split)
    d = (pl.concat([pl.scan_parquet(NORM / f"{prefix}source{i}.parquet").select("entity_id", "has_addr") for i in (2, 3)])
           .filter(~pl.col("has_addr")).select("entity_id").collect())
    if split in ("train", "eval"):
        keep = pl.read_parquet(NORM / ("train_queries.parquet" if split == "train" else "eval_queries.parquet"))
        d = d.join(keep, on="entity_id", how="semi")
    return d


def blank_addr(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns(pl.lit("").alias("addr_latin"), pl.lit("").alias("city"), pl.lit("").alias("state"))


def stage_sparse(split: str):
    t0 = time.time()
    src, _ = src_of(split)
    ids = no_addr_ids(split)
    s1 = blank_addr(load(src, 1))
    q = blank_addr(pl.concat([load(src, i) for i in (2, 3)]).join(ids, on="entity_id", how="semi"))
    print(f"{split}: {q.height:,} records without address ({time.time() - t0:.0f}s)", flush=True)
    SparseIndex(s1).search(q, top_k=DEPTH).write_parquet(NORM / "cand" / f"{split}_extra_sparse.parquet")
    print(f"sparse name-only done ({time.time() - t0:.0f}s)", flush=True)


def name_texts(prefix: str, n: int, ids: pl.DataFrame | None) -> pl.DataFrame:
    d = pl.read_parquet(NORM / f"{prefix}source{n}.parquet", columns=["entity_id", "country", "name_norm"])
    if ids is not None:
        d = d.join(ids, on="entity_id", how="semi")
    return d.select("entity_id", "country", pl.concat_str([pl.lit("query: "), pl.col("name_norm"), pl.lit(" | ")]).alias("text"))


def stage_dense(split: str):
    import embed  # torch: only in this process
    t0 = time.time()
    _, prefix = src_of(split)
    ids = no_addr_ids(split)
    s1t = name_texts(prefix, 1, None)
    qt = pl.concat([name_texts(prefix, i, ids) for i in (2, 3)])
    print(f"{split}: {qt.height:,} records without address ({time.time() - t0:.0f}s)", flush=True)
    d = embed.search(s1t, lambda c: [qt.filter(pl.col("country") == c)], embed.load_model(), top_k=DEPTH)
    d.write_parquet(NORM / "cand" / f"{split}_extra_dense.parquet")
    print(f"dense name-only done ({time.time() - t0:.0f}s)", flush=True)


def base_candidates(split: str, ids: pl.DataFrame) -> pl.DataFrame:
    """The regular top-10 sparse U top-10 dense pairs of these records (what the extras must not repeat)."""
    rec = ids.rename({"entity_id": "rec"}).lazy()
    if split == "eval":
        sp = pl.scan_parquet(NORM / "eval_sparse.parquet").filter(pl.col("sparse_rank") <= 10).select("rec", "s1")
        de = pl.scan_parquet(NORM / "eval_dense.parquet").filter(pl.col("dense_rank") <= 10).select("rec", "s1")
    else:
        sp = pl.scan_parquet(NORM / "cand" / f"{split}_sparse" / "*.parquet").select("rec", "s1")
        de = pl.scan_parquet(NORM / "cand" / f"{split}_dense" / "*.parquet").select("rec", "s1")
    return pl.concat([sp.join(rec, on="rec", how="semi"), de.join(rec, on="rec", how="semi")]).unique().collect()


def build(split: str):
    ids = no_addr_ids(split)
    legacy = split == "eval" and not (NORM / "cand" / "eval_extra_sparse.parquet").exists()
    if legacy:  # older local runs: channels of ablate_extra.py, computed for all held-out queries
        sp = pl.read_parquet(NORM / "eval_sparse_nameonly.parquet")
        dn = pl.read_parquet(NORM / "eval_dense_nameonly.parquet")
    else:
        sp = pl.read_parquet(NORM / "cand" / f"{split}_extra_sparse.parquet")
        dn = pl.read_parquet(NORM / "cand" / f"{split}_extra_dense.parquet")
    keep = ids.rename({"entity_id": "rec"})
    sp, dn = sp.join(keep, on="rec", how="semi"), dn.join(keep, on="rec", how="semi")
    base = base_candidates(split, ids).with_columns(pl.lit(1).alias("in_base"))
    f = pl.concat([dn.select("rec", "s1", (1.0 / (RRF + pl.col("dense_rank"))).alias("sc")),
                   sp.select("rec", "s1", (1.0 / (RRF + pl.col("sparse_rank"))).alias("sc"))]).group_by("rec", "s1").agg(pl.col("sc").sum())
    f = f.join(base, on=["rec", "s1"], how="left").filter(pl.col("in_base").is_null()).drop("in_base")
    f = f.sort("sc", descending=True).group_by("rec", maintain_order=True).head(N_EXTRA)
    f = f.with_columns(pl.int_range(1, pl.len() + 1).over("rec").cast(pl.Int8).alias("extra_rank")).select("rec", "s1", "extra_rank")
    out = NORM / "cand" / f"{split}_extra{EXTRA_TAG}.parquet"
    f.write_parquet(out)
    print(f"{split}: {f.height:,} extra candidates for {f['rec'].n_unique():,} of {ids.height:,} records without address -> {out}", flush=True)


if __name__ == "__main__":
    stage, split = sys.argv[1], sys.argv[2]
    (NORM / "cand").mkdir(exist_ok=True)
    {"sparse": stage_sparse, "dense": stage_dense, "build": build}[stage](split)
