"""Fuse the saved name-only variants of blocking_noaddr.py: reciprocal-rank fusion, and unions of per-variant top-N.
Recall of the true S1 owner among the top-10 fused, among the union of each variant's top-N (bigger lists, 'never miss'), and both
together with today's regular candidates.

  python blocking_noaddr_fuse.py char3_core word12_core [...]
"""
import sys

import polars as pl

import blocking_noaddr as B
from ranker import NORM

RRF = 60.0


def main():
    names = sys.argv[1:]
    s1 = pl.read_parquet(NORM / "source1.parquet", columns=["country", "name_core"])
    for split in ("eval", "train"):
        q = B.queries(split)
        own = q.filter(pl.col("s1").is_not_null()).select("rec", "s1")
        base = B.existing(split).join(q.select("rec"), on="rec", how="semi")
        c = {n: pl.read_parquet(NORM / "cand" / f"na_block_{n}_{split}.parquet") for n in names}
        print(f"[{split}] owned {own.height:,}; today's {base.height / q.height:.1f} cands/record recall "
              f"{own.join(base, on=['rec', 's1'], how='semi').height / own.height:.4f}", flush=True)
        f = (pl.concat([d.select("rec", "s1", (1.0 / (RRF + pl.col("rank"))).alias("sc")) for d in c.values()])
               .group_by("rec", "s1").agg(pl.col("sc").sum()).sort("sc", descending=True))
        f = f.with_columns(pl.int_range(1, pl.len() + 1).over("rec").alias("rank"))
        def rec(cand):
            return own.join(cand.select("rec", "s1").unique(), on=["rec", "s1"], how="semi").height / own.height, cand.select("rec", "s1").unique().height / q.height
        for k in (10, 20, 30):
            r, n = rec(f.filter(pl.col("rank") <= k))
            u = pl.concat([base, f.filter(pl.col("rank") <= k).select("rec", "s1")]).unique()
            ru = own.join(u, on=["rec", "s1"], how="semi").height / own.height
            print(f"  RRF top-{k:<3d}                    recall {r:.4f}  ({n:5.1f} cands/rec)   | + today's: {ru:.4f} ({u.height / q.height:.1f} cands/rec)", flush=True)
        for k in (5, 10, 15):
            u = pl.concat([d.filter(pl.col("rank") <= k).select("rec", "s1") for d in c.values()]).unique()
            r = own.join(u, on=["rec", "s1"], how="semi").height / own.height
            ub = pl.concat([base, u]).unique()
            print(f"  union of each variant's top-{k:<3d}  recall {r:.4f}  ({u.height / q.height:5.1f} cands/rec)   | + today's: {own.join(ub, on=['rec', 's1'], how='semi').height / own.height:.4f} ({ub.height / q.height:.1f} cands/rec)", flush=True)


if __name__ == "__main__":
    main()
