"""Pairwise ranker over the blocked candidates (sparse top-10 U dense top-10 per S2/S3 record).

Features are country-agnostic (no country one-hot, so France in test is handled like any other country):
  retrieval : sparse/dense score + rank, found-by-both, gap to the record's best candidate, margin best-vs-2nd
  name      : rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the latin core name,
              ratio on the native-script name, legal-form agreement, domain flag, lengths
  address   : token-set / partial on latin address, city / state / postal / first-house-number agreement,
              address-present flags

Usage:
  python ranker.py features train|eval|test     -> normalized/feat_<name>.parquet
  python ranker.py fit                          -> models/ranker.txt, prints held-out metrics
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

from block import NORM, ROOT, TOP_K, ground_truth

TEXT_COLS = ["entity_id", "name_core", "name_norm", "legal_form", "is_domain", "addr_latin", "city", "state", "postal", "has_addr"]
FEAT_CHUNK = 1_000_000
MODEL_PATH = ROOT / "models" / "ranker.txt"


def candidates(name: str) -> pl.DataFrame:
    """Union of both channels' top-K for a candidate set: 'train', 'test' (parts dirs) or 'eval' (held-out files)."""
    if name == "eval":
        sp = pl.read_parquet(NORM / "eval_sparse.parquet").filter(pl.col("sparse_rank") <= TOP_K)
        de = pl.read_parquet(NORM / "eval_dense.parquet").filter(pl.col("dense_rank") <= TOP_K)
    else:
        sp = pl.read_parquet(NORM / "cand" / f"{name}_sparse" / "*.parquet")
        de = pl.read_parquet(NORM / "cand" / f"{name}_dense" / "*.parquet")
    return merge_channels(sp, de)


def merge_channels(sp: pl.DataFrame, de: pl.DataFrame) -> pl.DataFrame:
    c = sp.join(de, on=["rec", "s1"], how="full", coalesce=True)
    return c.with_columns(
        pl.col("sparse_score").fill_null(0.0), pl.col("sparse_rank").fill_null(99).cast(pl.Int16),
        pl.col("dense_score").fill_null(-1.0), pl.col("dense_rank").fill_null(99).cast(pl.Int16))


def retrieval_features(c: pl.DataFrame) -> pl.DataFrame:
    """Per-record context: how this candidate compares with the record's other candidates."""
    ds, ss = pl.col("dense_score"), pl.col("sparse_score")
    c = c.with_columns(
        ((pl.col("sparse_rank") < 99) & (pl.col("dense_rank") < 99)).alias("both"),
        (ds.max().over("rec") - ds).alias("dense_gap_best"),
        (ss.max().over("rec") - ss).alias("sparse_gap_best"),
        pl.len().over("rec").alias("n_cands"),
        ds.rank("ordinal", descending=True).over("rec").cast(pl.Int16).alias("dense_rank_u"),
    )
    second = (c.filter(pl.col("dense_rank_u") == 2).select("rec", pl.col("dense_score").alias("_d2")))
    c = c.join(second, on="rec", how="left").with_columns(
        pl.when(pl.col("dense_rank_u") == 1).then(pl.col("dense_score") - pl.col("_d2").fill_null(-1.0))
          .otherwise(pl.col("dense_score") - pl.col("dense_score").max().over("rec")).alias("dense_margin")).drop("_d2")
    return c


def _sim(scorer, a: list, b: list) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def string_features(c: pl.DataFrame, texts: pl.DataFrame) -> pl.DataFrame:
    t = texts
    c = (c.join(t.rename({k: f"q_{k}" for k in t.columns}), left_on="rec", right_on="q_entity_id", how="left")
          .join(t.rename({k: f"s_{k}" for k in t.columns}), left_on="s1", right_on="s_entity_id", how="left"))
    qn, sn = c["q_name_core"].fill_null("").to_list(), c["s_name_core"].fill_null("").to_list()
    qa, sa = c["q_addr_latin"].fill_null("").to_list(), c["s_addr_latin"].fill_null("").to_list()
    feats = {
        "name_ratio": _sim(fuzz.ratio, qn, sn),
        "name_tset": _sim(fuzz.token_set_ratio, qn, sn),
        "name_tsort": _sim(fuzz.token_sort_ratio, qn, sn),
        "name_partial": _sim(fuzz.partial_ratio, qn, sn),
        "name_jw": _sim(distance.JaroWinkler.normalized_similarity, qn, sn),
        "name_native_ratio": _sim(fuzz.ratio, c["q_name_norm"].fill_null("").to_list(), c["s_name_norm"].fill_null("").to_list()),
        "addr_tset": _sim(fuzz.token_set_ratio, qa, sa),
        "addr_partial": _sim(fuzz.partial_ratio, qa, sa),
    }
    c = c.with_columns([pl.Series(k, v) for k, v in feats.items()])

    def eq(col):  # 1 equal, 0 different, -1 unknown on either side
        a, b = pl.col(f"q_{col}").fill_null(""), pl.col(f"s_{col}").fill_null("")
        return pl.when((a == "") | (b == "")).then(-1).when(a == b).then(1).otherwise(0).cast(pl.Int8).alias(f"{col}_eq")

    first_num = lambda s: pl.col(s).fill_null("").str.extract(r"(\d+)").str.strip_chars_start("0")
    return c.with_columns(
        eq("city"), eq("state"), eq("postal"), eq("legal_form"),
        pl.when(first_num("q_addr_latin").is_null() | first_num("s_addr_latin").is_null()).then(-1)
          .when(first_num("q_addr_latin") == first_num("s_addr_latin")).then(1).otherwise(0).cast(pl.Int8).alias("num_eq"),
        (pl.col("q_is_domain") | pl.col("s_is_domain")).cast(pl.Int8).alias("any_domain"),
        pl.col("q_has_addr").cast(pl.Int8).alias("q_has_addr"), pl.col("s_has_addr").cast(pl.Int8).alias("s_has_addr"),
        pl.col("q_name_core").str.len_chars().alias("q_name_len"), pl.col("s_name_core").str.len_chars().alias("s_name_len"),
        pl.col("rec").str.starts_with("S3").cast(pl.Int8).alias("from_s3"),
    ).drop([x for x in c.columns if x.startswith(("q_", "s_")) and x not in ("q_has_addr", "s_has_addr")])


FEATURES = ["sparse_score", "sparse_rank", "dense_score", "dense_rank", "both", "dense_gap_best", "sparse_gap_best",
            "n_cands", "dense_margin", "name_ratio", "name_tset", "name_tsort", "name_partial",
            "name_jw", "name_native_ratio", "addr_tset", "addr_partial", "city_eq", "state_eq", "postal_eq",
            "legal_form_eq", "num_eq", "any_domain", "q_has_addr", "s_has_addr", "q_name_len", "s_name_len", "from_s3"]


def build_features(name: str):
    t0 = time.time()
    split = "test" if name == "test" else "train"
    prefix = "" if split == "train" else "test_"
    c = retrieval_features(candidates(name))
    print(f"{name}: {c.height:,} pairs, {c['rec'].n_unique():,} records ({time.time() - t0:.0f}s)", flush=True)
    need = pl.concat([c.select(pl.col("rec").alias("entity_id")), c.select(pl.col("s1").alias("entity_id"))]).unique()
    texts = pl.concat([pl.read_parquet(NORM / f"{prefix}source{i}.parquet", columns=TEXT_COLS).join(need, on="entity_id", how="semi")
                       for i in (1, 2, 3)])  # one source at a time, keep only the rows the candidates touch
    parts = []
    for i in range(0, c.height, FEAT_CHUNK):
        parts.append(string_features(c.slice(i, FEAT_CHUNK), texts))
        print(f"  features {min(i + FEAT_CHUNK, c.height):,}/{c.height:,} ({time.time() - t0:.0f}s)", flush=True)
    f = pl.concat(parts)
    if split == "train":
        f = f.join(ground_truth().with_columns(pl.lit(1, pl.Int8).alias("label")), on=["rec", "s1"], how="left") \
             .with_columns(pl.col("label").fill_null(0))
    f.write_parquet(NORM / f"feat_{name}.parquet")
    print(f"saved feat_{name} ({time.time() - t0:.0f}s)", flush=True)


def assign(f: pl.DataFrame, p: np.ndarray, thr: float) -> pl.DataFrame:
    """One owner per record: keep the record's best candidate if its probability clears the threshold."""
    return (f.select("rec", "s1").with_columns(pl.Series("p", p))
             .sort("p", descending=True).group_by("rec", maintain_order=True).head(1).filter(pl.col("p") >= thr))


def fit(max_rows: int | None = None, out: Path = MODEL_PATH):
    import gc
    import lightgbm as lgb
    tr = pl.read_parquet(NORM / "feat_train.parquet", n_rows=max_rows)
    ev = pl.read_parquet(NORM / "feat_eval.parquet", n_rows=max_rows)
    print(f"train pairs {tr.height:,} (pos {tr['label'].sum():,})   eval pairs {ev.height:,} (pos {ev['label'].sum():,})", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=14)
    # float32 matrix (float64 would need ~4 GB for 17M x 28) and drop the frame once the binned Dataset exists
    dtr = lgb.Dataset(tr.select(FEATURES).cast(pl.Float32).to_numpy(), tr["label"].to_numpy(),
                      feature_name=FEATURES, params={"max_bin": 255})
    dtr.construct()
    del tr
    gc.collect()
    xe = ev.select(FEATURES).cast(pl.Float32).to_numpy()
    dev = lgb.Dataset(xe, ev["label"].to_numpy(), reference=dtr)
    m = lgb.train(params, dtr, 2000, valid_sets=[dev], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(str(out))
    p = m.predict(xe)
    truth = ev.filter(pl.col("label") == 1).height  # owned records whose owner is among the candidates
    owned = ground_truth().join(pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"}), on="rec", how="semi")
    owned = owned.filter(pl.col("s1").hash(seed=7) % 10 == 0).height
    print(f"blocking recall on eval: {truth / owned:.4f}", flush=True)
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        a = assign(ev, p, thr).join(ev.select("rec", "s1", "label"), on=["rec", "s1"])
        prec, rec = a["label"].mean(), a["label"].sum() / owned
        f05 = 1.25 * prec * rec / (0.25 * prec + rec)
        print(f"  thr {thr:.2f}: pair precision {prec:.4f}  recall {rec:.4f}  F0.5 {f05:.4f}", flush=True)
    imp = sorted(zip(FEATURES, m.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", ", ".join(f"{k}" for k, _ in imp[:12]))


if __name__ == "__main__":
    if sys.argv[1] == "features":
        build_features(sys.argv[2])
    elif sys.argv[1] == "fit":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None  # a row limit means smoke test: never overwrite the real model
        fit(n, MODEL_PATH if n is None else MODEL_PATH.with_name("ranker_smoketest.txt"))
