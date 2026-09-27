"""Why is B2 unsure on French records? Per-feature contributions (LightGBM pred_contrib, i.e. SHAP) of the best candidate pair of a record,
French vs US / India, on real test candidates (one scoring chunk; nothing is fitted). Also compares the raw feature values.

  ER_CE=1 ER_CE_TAG=_v2 python france_shap.py [chunk ...]
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import predict
import ranker
from block import CHUNK, NORM


def chunk_features(n: int, ids, names, dense, dense_rec, extras, texts, ce, n_s1) -> pl.DataFrame:
    lo, hi = n_s1 + n * CHUNK, n_s1 + (n + 1) * CHUNK
    a, b = dense_rec.search_sorted(lo), dense_rec.search_sorted(hi)
    de = dense.slice(a, b - a)
    de = de.select(pl.Series("rec", names.gather(de["rec_i"].to_numpy())), pl.Series("s1", names.gather(de["s1_i"].to_numpy())), "dense_score", "dense_rank")
    sp = pl.read_parquet(NORM / "cand" / "test_sparse" / f"part{n:04d}.parquet")
    c = ranker.merge_channels(sp, de)
    ex = None if extras is None else extras.join(c.select("rec").unique(), on="rec", how="semi")
    c = ranker.retrieval_features(ranker.add_extras(c, ex))
    need = pl.concat([c.select(pl.col("rec").alias("entity_id")), c.select(pl.col("s1").alias("entity_id"))]).unique()
    return ranker.join_ce(ranker.string_features(c, texts.join(need, on="entity_id", how="semi")), ce)


def main():
    chunks = [int(x) for x in sys.argv[1:]] or [0, 60, 120]
    ids = predict.ids_table(); names = ids["entity_id"]
    n_s1 = pl.scan_parquet(NORM / "test_source1.parquet").select(pl.len()).collect().item()
    texts = pl.concat([ranker.read_texts("test_", i) for i in (1, 2, 3)])
    dense = predict.load_dense(ids); dense_rec = dense["rec_i"]
    extras = ranker.load_extras("test")
    ce = pl.read_parquet(NORM / "ce_test_v2.parquet")
    country = pl.concat([pl.read_parquet(NORM / f"test_source{i}.parquet", columns=["entity_id", "country"]) for i in (2, 3)]).rename({"entity_id": "rec"})
    b2 = lgb.Booster(model_file=str(ranker.ROOT / "models" / "ranker_b2.txt")); cols = b2.feature_name()
    f = pl.concat([chunk_features(n, ids, names, dense, dense_rec, extras, texts, ce, n_s1) for n in chunks])
    x = f.select(cols).cast(pl.Float32).to_numpy()
    p = b2.predict(x)
    f = f.with_columns(pl.Series("p", p)).with_row_index("i").join(country, on="rec")
    top = f.sort("p", descending=True).group_by("rec", maintain_order=True).head(1)
    contrib = b2.predict(x[top["i"].to_numpy()], pred_contrib=True)   # (n, n_features + 1), log-odds
    C = pl.DataFrame(contrib[:, :-1], schema=cols).with_columns(pl.Series("country", top["country"]), pl.Series("p", top["p"]))
    unsure = (pl.col("p") >= 0.05) & (pl.col("p") < 0.98)
    for band, flt in (("UNSURE best pairs (0.05-0.98)", unsure), ("CONFIDENT best pairs (>= 0.98)", pl.col("p") >= 0.98)):
        g = C.filter(flt).group_by("country").agg(pl.len().alias("n"), *[pl.col(c).mean() for c in cols])
        fr = g.filter(pl.col("country") == "France"); ot = g.filter(pl.col("country") != "France")
        if fr.height == 0 or ot.height == 0:
            continue
        diff = {c: float(fr[c][0]) - float(ot[c].mean()) for c in cols}
        print(f"\n=== {band}: France n={int(fr['n'][0])}, US/India n={int(ot['n'].sum())}")
        print("features pushing French pairs DOWN relative to US/India (mean log-odds difference):")
        for c, v in sorted(diff.items(), key=lambda t: t[1])[:12]:
            print(f"  {c:22s} {v:+.3f}")
        print("features pushing French pairs UP:")
        for c, v in sorted(diff.items(), key=lambda t: -t[1])[:6]:
            print(f"  {c:22s} {v:+.3f}")
    V = top.filter(unsure)
    key = ["dense_score", "dense_margin", "dense_rank", "sparse_score", "sparse_rank", "both", "ce_score", "name_ratio", "name_tset", "addr_tset",
           "num_equal", "num_conflict", "rare_only_q", "rare_only_s", "gen_only_q", "legal_same", "legal_missing_one", "n_cands"]
    print("\n=== raw feature means of UNSURE best pairs by country")
    print(V.group_by("country").agg(pl.len().alias("n"), *[pl.col(k).cast(pl.Float64).mean().round(3) for k in key]).sort("country").transpose(include_header=True))


if __name__ == "__main__":
    main()
