"""Does a name-only slot group (type A: record without address) and an address-only slot group (type B: unrelated
name, matching address) lift the recall of the top-10 sparse U top-10 dense candidate set?

  python ablate_extra.py          held-out queries, full 2.2M S1 index
"""
import sys
import time

import polars as pl

from block import NORM, SparseIndex, ground_truth, is_val_s1, load

DEPTH = 30
N_EXTRA = 5
RRF = 60.0


def texts_variant(n: int, qid: pl.DataFrame | None, mode: str) -> pl.DataFrame:
    """entity_id, country, text for source n; mode 'name' or 'addr' blanks the other field."""
    d = pl.read_parquet(NORM / f"source{n}.parquet", columns=["entity_id", "country", "name_norm", "addr_norm"])
    if qid is not None:
        d = d.join(qid, on="entity_id", how="semi")
    name = pl.col("name_norm") if mode == "name" else pl.lit("")
    addr = pl.col("addr_norm") if mode == "addr" else pl.lit("")
    return d.select("entity_id", "country", pl.concat_str([pl.lit("query: "), name, pl.lit(" | "), addr]).alias("text"))


def rrf_extra(chan_a: pl.DataFrame, chan_b: pl.DataFrame, base: pl.DataFrame, n_extra: int) -> pl.DataFrame:
    """Reciprocal-rank fusion of two channels; the n_extra best candidates that are not already in `base`."""
    a = chan_a.select("rec", "s1", (1.0 / (RRF + pl.col("dense_rank"))).alias("sc"))
    b = chan_b.select("rec", "s1", (1.0 / (RRF + pl.col("sparse_rank"))).alias("sc"))
    f = pl.concat([a, b]).group_by("rec", "s1").agg(pl.col("sc").sum())
    f = f.join(base.select("rec", "s1").with_columns(pl.lit(1).alias("in_base")), on=["rec", "s1"], how="left").filter(pl.col("in_base").is_null())
    f = f.sort("sc", descending=True).group_by("rec", maintain_order=True).head(n_extra)
    return f.select("rec", "s1")


def hit(u: pl.DataFrame, truth: pl.DataFrame) -> float:
    return truth.join(u, on=["rec", "s1"], how="semi").height / truth.height


def stage_sparse(qid):
    """address-only sparse channel (polars only; torch is never imported in this process)"""
    t0 = time.time()
    s1 = load("train", 1)
    q = pl.concat([load("train", i) for i in (2, 3)]).join(qid, on="entity_id", how="semi")
    blank_name = lambda d: d.with_columns(pl.lit("").alias("name_core"))
    SparseIndex(blank_name(s1)).search(blank_name(q), top_k=DEPTH).write_parquet(NORM / "eval_sparse_addronly.parquet")
    print(f"sparse address-only done ({time.time() - t0:.0f}s)", flush=True)


def stage_dense(qid):
    """dense name-only and address-only channels (GPU; polars is used only for light frame handling)"""
    import embed
    t0 = time.time()
    model = embed.load_model()
    for mode in ("name", "addr"):
        s1t = texts_variant(1, None, mode)
        qt = pl.concat([texts_variant(i, qid, mode) for i in (2, 3)])
        d = embed.search(s1t, lambda c, qt=qt: [qt.filter(pl.col("country") == c)], model, top_k=DEPTH)
        d.write_parquet(NORM / f"eval_dense_{mode}only.parquet")
        print(f"dense {mode}-only done ({time.time() - t0:.0f}s)", flush=True)
        del s1t, qt, d


def stage_report(qid, truth):
    has_addr = pl.concat([pl.read_parquet(NORM / f"source{i}.parquet", columns=["entity_id", "has_addr"]) for i in (2, 3)])
    has_addr = has_addr.join(qid, on="entity_id", how="semi").rename({"entity_id": "rec"})
    sp = pl.read_parquet(NORM / "eval_sparse.parquet")
    de = pl.read_parquet(NORM / "eval_dense_top100.parquet")
    base = pl.concat([sp.filter(pl.col("sparse_rank") <= 10).select("rec", "s1"),
                      de.filter(pl.col("dense_rank") <= 10).select("rec", "s1")]).unique()
    sp_name = pl.read_parquet(NORM / "eval_sparse_nameonly.parquet")
    sp_addr = pl.read_parquet(NORM / "eval_sparse_addronly.parquet")
    dn = pl.read_parquet(NORM / "eval_dense_nameonly.parquet")
    da = pl.read_parquet(NORM / "eval_dense_addronly.parquet")
    xa = rrf_extra(dn, sp_name, base, N_EXTRA)      # best name-only candidates not already present
    xb = rrf_extra(da, sp_addr, base, N_EXTRA)      # best address-only candidates not already present
    noaddr = has_addr.filter(~pl.col("has_addr")).select("rec")
    withaddr = has_addr.filter(pl.col("has_addr")).select("rec")
    nq = qid.height

    def show(label, u):
        print(f"{label:66s} recall {hit(u, truth) * 100:6.2f}%   candidates/query {u.height / nq:5.1f}", flush=True)

    print(f"\nheld-out: {nq:,} queries, {truth.height:,} true owners; {noaddr.height:,} queries have no address")
    u = lambda *parts: pl.concat(list(parts)).select("rec", "s1").unique()
    show("current: top-10 sparse U top-10 dense", base)
    show("+ 5 name-only extras for ALL records", u(base, xa))
    show("+ 5 address-only extras for ALL records", u(base, xb))
    show("+ both (5 + 5) for ALL records   [your plan]", u(base, xa, xb))
    show("+ name-only 5 only for records WITHOUT an address", u(base, xa.join(noaddr, on="rec", how="semi")))
    show("+ name-only 5 (no-address records) + address-only 5 (all)", u(base, xa.join(noaddr, on="rec", how="semi"), xb))
    show("+ name-only 5 (no-address records) + address-only 5 (with-address records)",
         u(base, xa.join(noaddr, on="rec", how="semi"), xb.join(withaddr, on="rec", how="semi")))
    show("for comparison: + dense ranks 11-15 and sparse ranks 11-15",
         u(base, de.filter((pl.col("dense_rank") > 10) & (pl.col("dense_rank") <= 15)).select("rec", "s1"),
           sp.filter((pl.col("sparse_rank") > 10) & (pl.col("sparse_rank") <= 15)).select("rec", "s1")))
    show("for comparison: + dense ranks 11-20", u(base, de.filter((pl.col("dense_rank") > 10) & (pl.col("dense_rank") <= 20)).select("rec", "s1")))
    miss = truth.join(base, on=["rec", "s1"], how="anti").join(has_addr, on="rec")
    print(f"\ncurrently missed: {miss.height} ({(~miss['has_addr']).sum()} without address, {miss['has_addr'].sum()} with address)")
    for label, x in (("name-only extras", xa), ("address-only extras", xb), ("both", u(xa, xb))):
        r = miss.join(x, on=["rec", "s1"], how="semi")
        print(f"  {label:20s} recovers {r.height:>3} of {miss.height}   ({(~r['has_addr']).sum()} of them records without address)")


if __name__ == "__main__":
    stage = sys.argv[1]
    qid = pl.read_parquet(NORM / "eval_queries.parquet")
    truth = ground_truth().filter(is_val_s1()).join(qid.rename({"entity_id": "rec"}), on="rec", how="semi")
    {"sparse": lambda: stage_sparse(qid), "dense": lambda: stage_dense(qid), "report": lambda: stage_report(qid, truth)}[stage]()
