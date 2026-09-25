"""Recall of the true owner at depth k (10..100) for sparse / dense, with and without the address.

  python ablate_depth.py        -> prints a table; saves normalized/ablate_depth.csv
Uses the same held-out queries as `block.py eval`, against the full 2.2M-entity S1 index.
"""
import sys
import time

import polars as pl

import embed
from block import NORM, SparseIndex, ground_truth, is_val_s1, load

KS = (1, 5, 10, 20, 30, 60, 80, 100)
DEPTH = 100


def recall_at(cand: pl.DataFrame, truth: pl.DataFrame, rank_col: str) -> dict:
    r = truth.join(cand.select("rec", "s1", rank_col), on=["rec", "s1"], how="left")[rank_col].fill_null(10_000)
    return {k: float((r <= k).mean()) for k in KS}


if __name__ == "__main__":
    t0 = time.time()
    qid = pl.read_parquet(NORM / "eval_queries.parquet")
    truth = ground_truth().filter(is_val_s1()).join(qid.rename({"entity_id": "rec"}), on="rec", how="semi")
    blank = lambda d: d.with_columns(pl.lit("").alias("addr_latin"), pl.lit("").alias("city"), pl.lit("").alias("state"))
    rows = {}

    # ---- sparse  (`python ablate_depth.py dense` skips this part and reuses the numbers from a previous run)
    if len(sys.argv) > 1 and sys.argv[1] == "dense":
        rows["sparse  + address"] = {1: 0.8969, 5: 0.9345, 10: 0.9441, 20: 0.9516, 30: 0.9558, 60: 0.9621, 80: 0.965, 100: 0.967}
        rows["sparse  name only"] = {1: 0.3706, 5: 0.544, 10: 0.5939, 20: 0.6361, 30: 0.661, 60: 0.7039, 80: 0.7184, 100: 0.7284}
    else:
        s1 = load("train", 1)
        q = pl.concat([load("train", i) for i in (2, 3)]).join(qid, on="entity_id", how="semi")
        for name, f in (("sparse  + address", lambda d: d), ("sparse  name only", blank)):
            cand = SparseIndex(f(s1)).search(f(q), top_k=DEPTH)
            rows[name] = recall_at(cand, truth, "sparse_rank")
            print(name, {k: round(v, 4) for k, v in rows[name].items()}, f"({time.time() - t0:.0f}s)", flush=True)
            del cand
        del s1, q

    # ---- dense (fine-tuned e5). name-only = same text format with an empty address
    model = embed.load_model()
    t1 = embed.texts("train", 1)
    tq = pl.concat([embed.texts("train", i).join(qid, on="entity_id", how="semi") for i in (2, 3)])
    def name_only(d: pl.DataFrame, n: int) -> pl.DataFrame:
        src = pl.read_parquet(NORM / ("source1.parquet" if n == 1 else f"source{n}.parquet"), columns=["entity_id", "name_norm"])
        return d.drop("text").join(src, on="entity_id").with_columns(
            pl.concat_str([pl.lit("query: "), pl.col("name_norm"), pl.lit(" | ")]).alias("text")).drop("name_norm")
    for name, s1t, qt in (("dense   + address", t1, tq),
                          ("dense   name only", name_only(t1, 1), pl.concat([name_only(embed.texts("train", i).join(qid, on="entity_id", how="semi"), i) for i in (2, 3)]))):
        cand = embed.search(s1t, lambda c, qt=qt: [qt.filter(pl.col("country") == c)], model, top_k=DEPTH)
        rows[name] = recall_at(cand, truth, "dense_rank")
        print(name, {k: round(v, 4) for k, v in rows[name].items()}, f"({time.time() - t0:.0f}s)", flush=True)
        if name.endswith("+ address"):
            cand.write_parquet(NORM / "eval_dense_top100.parquet")
        del cand

    print("\nrecall of the true owner within the top-k candidates (held-out, full S1 index)")
    print(f"{'':20s}" + "".join(f"{'@' + str(k):>9s}" for k in KS))
    for name, r in rows.items():
        print(f"{name:20s}" + "".join(f"{r[k] * 100:>8.2f}%" for k in KS))
    pl.DataFrame([{"channel": n, **{f"@{k}": v for k, v in r.items()}} for n, r in rows.items()]).write_csv(NORM / "ablate_depth.csv")
