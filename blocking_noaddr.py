"""Experiment: simple name-only blocking (TF-IDF over character n-grams / words, cosine, GPU) for the records WITHOUT an address.

No model is trained. Every variant returns the top-K S1 candidates of a record inside its own country; the report gives the recall of the
true S1 owner at K = 1, 5, 10, 20, 50, 100 on the held-out no-address queries (eval) and on the ranker-training no-address queries
(bigger sample, same procedure), next to what the current pipeline finds, and the recall of the UNION with the current top-10 sparse + top-10 dense.

  python blocking_noaddr.py build train|eval|test   the recommended candidates for a split -> normalized/cand/na_tfidf_<split>.parquet
  python blocking_noaddr.py [variants]  run the variants (ER_NA_SAMPLE_MOD=5: every 5th train query); adaptive() is the recommended candidate list
  python blocking_noaddr.py            run all variants, write normalized/cand/na_block_<variant>_<split>.parquet (rec, s1, rank<=K)
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from block import ground_truth
from ranker import NORM

K = 100
BATCH = 256
VARIANTS = {   # name -> (text column, TfidfVectorizer kwargs)
    "char3_core":  ("name_core", dict(analyzer="char_wb", ngram_range=(3, 3))),
    "char2-4_core": ("name_core", dict(analyzer="char_wb", ngram_range=(2, 4))),
    "char4_core":  ("name_core", dict(analyzer="char_wb", ngram_range=(4, 4))),
    "char3_norm":  ("name_norm", dict(analyzer="char_wb", ngram_range=(3, 3))),
    "char2-4_norm": ("name_norm", dict(analyzer="char_wb", ngram_range=(2, 4))),
    "char3x_norm": ("name_norm", dict(analyzer="char", ngram_range=(3, 3))),     # n-grams may span word boundaries
    "char4_norm":  ("name_norm", dict(analyzer="char_wb", ngram_range=(4, 4))),
    "word12_norm": ("name_norm", dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")),
    "word12_core": ("name_core", dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")),
}


def queries(split: str) -> pl.DataFrame:
    """No-address S2/S3 records of a query set with their true S1 owner (null = orphan)."""
    prefix = "test_" if split == "test" else ""
    q = (pl.concat([pl.read_parquet(NORM / f"{prefix}source{i}.parquet", columns=["entity_id", "country", "name_core", "name_norm", "has_addr"]) for i in (2, 3)])
           .filter(~pl.col("has_addr")))
    if split != "test":  # eval / ranker-train / trainall queries are subsets of the train files
        q = q.join(pl.read_parquet(NORM / f"{split}_queries.parquet"), on="entity_id", how="semi")
    q = q.rename({"entity_id": "rec"})
    if split == "test":
        return q.with_columns(pl.lit(None, pl.Utf8).alias("s1"))
    mod = int(os.environ.get("ER_NA_SAMPLE_MOD", 1)) if split == "train" else 1  # e.g. 5: every 5th train query (the full 31k is slow for the big variants)
    if mod > 1:
        q = q.filter(pl.col("rec").hash(seed=21) % mod == 0)
    return q.join(ground_truth(), on="rec", how="left")


def search(s1: pl.DataFrame, q: pl.DataFrame, col: str, kw: dict) -> pl.DataFrame:
    """(rec, s1, rank) top-K per query, scoring inside each country on the GPU."""
    out = []
    for cty in q["country"].unique().to_list():
        s = s1.filter(pl.col("country") == cty)
        qc = q.filter(pl.col("country") == cty)
        vec = TfidfVectorizer(sublinear_tf=True, lowercase=False, dtype=np.float32, **kw)
        X = vec.fit_transform(s[col].fill_null("").to_list()).tocsr()
        Q = vec.transform(qc[col].fill_null("").to_list()).tocsr()
        Xt = torch.sparse_csr_tensor(torch.from_numpy(X.indptr).long(), torch.from_numpy(X.indices).long(), torch.from_numpy(X.data), size=X.shape).cuda()
        s1_ids = s["entity_id"].to_numpy()
        recs = qc["rec"].to_numpy()
        for b in range(0, Q.shape[0], BATCH):
            qd = torch.from_numpy(Q[b:b + BATCH].toarray()).cuda()
            sc = torch.sparse.mm(Xt, qd.T)                       # (n_s1, batch)
            top = torch.topk(sc, min(K, sc.shape[0]), dim=0)
            idx, val = top.indices.T.cpu().numpy(), top.values.T.cpu().numpy()
            keep = val > 0
            rows = np.repeat(np.arange(idx.shape[0]), keep.sum(1))
            out.append(pl.DataFrame({"rec": recs[b + rows], "s1": s1_ids[idx[keep]], "rank": np.tile(np.arange(1, idx.shape[1] + 1, dtype=np.int32), (idx.shape[0], 1))[keep], "score": val[keep]}))
        del Xt
        torch.cuda.empty_cache()
    return pl.concat(out)


def adaptive(cand: pl.DataFrame, frac: float = 0.85, cap: int = 100, base_k: int = 10) -> pl.DataFrame:
    """The recommended list: the top-10 plus every candidate scoring >= frac * the record's best score, at most `cap` per record.
    A fixed top-10 cuts a cluster of same-name S1 twins at an arbitrary point; this keeps the whole cluster (and near-ties) instead.
    Held-out no-address recall: 83.5% today (17 cands) -> ~93.8% with char 3-grams on name_norm at 0.85 / 100 (23.6 cands on average)."""
    top = cand.filter(pl.col("rank") == 1).select("rec", pl.col("score").alias("best"))
    c = cand.join(top, on="rec", how="left")
    return c.filter((pl.col("rank") <= base_k) | ((pl.col("score") >= frac * pl.col("best")) & (pl.col("rank") <= cap))).select("rec", "s1", "rank", "score")


def existing(split: str) -> pl.DataFrame:
    """The regular top-10 sparse U top-10 dense candidates the pipeline has today for these records (no name-only extras)."""
    f = pl.read_parquet(NORM / f"feat_{split}_ce2.parquet", columns=["rec", "s1", "q_has_addr", "extra_rank"])
    return f.filter((pl.col("q_has_addr") == 0) & (pl.col("extra_rank") == 0)).select("rec", "s1")


def report(name: str, cand: pl.DataFrame, own: pl.DataFrame, base: pl.DataFrame, twins: pl.DataFrame):
    t = own.join(cand, on=["rec", "s1"], how="left").with_columns(pl.col("rank").fill_null(10_000))
    r = t["rank"].to_numpy()
    line = "  ".join(f"@{k}:{(r <= k).mean():.4f}" for k in (1, 5, 10, 20, 50, 100))
    u = pl.concat([base, cand.filter(pl.col("rank") <= 10).select("rec", "s1")]).unique()
    un = own.join(u, on=["rec", "s1"], how="semi").height / own.height
    # records whose exact name has > 10 S1 twins cannot be resolved by any top-10: report recall on the others
    easy = t.join(twins, on="rec")
    e = easy.filter(pl.col("twins") <= 10)
    print(f"  {name:14s} {line}   | union with today's 17: {un:.4f}   | @10 without >10-twin records: {(e['rank'] <= 10).mean():.4f}", flush=True)
    return r


def build(split: str):
    """The recommended candidates (adaptive(): top-10 + near-ties, cap 100) of every no-address record of a split -> normalized/cand/na_tfidf_<split>.parquet.
    train / eval use the train S1 file (the full S1 index), test the test S1 file. ER_NOADDR_LIMIT=N: only the first N records (smoke test)."""
    t0 = time.time()
    s1 = pl.read_parquet(NORM / ("test_source1.parquet" if split == "test" else "source1.parquet"), columns=["entity_id", "country", "name_core", "name_norm"])
    q = queries(split)
    lim = int(os.environ.get("ER_NOADDR_LIMIT", 0))
    if lim:
        q = q.head(lim)
    kw = VARIANTS["char3x_norm"]
    cand = adaptive(search(s1, q, kw[0], kw[1]))
    out = NORM / "cand" / f"na_tfidf_{split}{'_smoke' if lim else ''}.parquet"
    cand.write_parquet(out)
    print(f"{split}: {q.height:,} no-address records, {cand.height / q.height:.1f} candidates each -> {out.name} ({time.time() - t0:.0f}s)", flush=True)


def main():
    t0 = time.time()
    s1 = pl.read_parquet(NORM / "source1.parquet", columns=["entity_id", "country", "name_core", "name_norm"])
    tw_tab = s1.group_by("country", "name_core").agg(pl.len().alias("twins"))
    variants = sys.argv[1:] or list(VARIANTS)
    for split in ("eval", "train"):
        q = queries(split)
        own = q.filter(pl.col("s1").is_not_null()).select("rec", "s1")
        twins = q.join(tw_tab, on=["country", "name_core"], how="left").select("rec", pl.col("twins").fill_null(0))
        base = existing(split).join(q.select("rec"), on="rec", how="semi")
        print(f"[{split}] {q.height:,} no-address queries, {own.height:,} owned", flush=True)
        b = own.join(base.with_columns(pl.lit(1).alias("rank")), on=["rec", "s1"], how="left")
        print(f"  today (top-10 sparse U top-10 dense, {base.height / q.height:.1f} cands/record): recall {b['rank'].is_not_null().mean():.4f}", flush=True)
        for v in variants:
            col, kw = VARIANTS[v]
            cand = search(s1, q, col, kw)
            cand.write_parquet(NORM / "cand" / f"na_block_{v}_{split}.parquet")
            report(v, cand, own, base, twins)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        sys.exit(main())
