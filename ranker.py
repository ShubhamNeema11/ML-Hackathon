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
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

import structfeat
from block import NORM, ROOT, TOP_K, ground_truth

TEXT_COLS = ["entity_id", "name_core", "name_norm", "legal_form", "is_domain", "addr_latin", "city", "state", "postal", "has_addr"]
FEAT_CHUNK = 1_000_000
MODEL_PATH = ROOT / "models" / os.environ.get("ER_MODEL", "ranker.txt")
CE_TAG = os.environ.get("ER_CE_TAG", "")  # suffix of the cross-encoder score files (ce_<split><tag>.parquet)
CE2_TAG = os.environ.get("ER_CE2_TAG", "")  # optional second cross-encoder (e.g. "_big"): ce2_score / ce2_gap_best next to the first one
FEAT_TAG = os.environ.get("ER_FEAT_TAG", "")  # e.g. "_v2": feature files feat_train_v2.parquet, kept apart from v1
# Telangana was split from Andhra Pradesh in 2014 and the sources disagree on Hyderabad-area addresses:
# 177 of the 245 true pairs with different states are exactly AP vs TS, so the two compare as the same state.
STATE_CANON = {"ts": "ap"}


def read_texts(prefix: str, i: int) -> pl.DataFrame:
    """Text columns of one source, with the state overlay from fix_states.py applied when it exists."""
    d = pl.read_parquet(NORM / f"{prefix}source{i}.parquet", columns=TEXT_COLS)
    fix = NORM / f"state_fix_{prefix}source{i}.parquet"
    if os.environ.get("ER_STATE_FIX", "1") == "1" and fix.exists():
        d = d.drop("state").join(pl.read_parquet(fix), on="entity_id", how="left").with_columns(pl.col("state").fill_null(""))
    return d.with_columns(pl.col("state").replace(STATE_CANON))


def candidates(name: str) -> pl.DataFrame:
    """Union of both channels' top-K for a candidate set: 'train', 'test' (parts dirs) or 'eval' (held-out files)."""
    if name == "eval":
        sp = pl.read_parquet(NORM / "eval_sparse.parquet").filter(pl.col("sparse_rank") <= TOP_K)
        de = pl.read_parquet(NORM / "eval_dense.parquet").filter(pl.col("dense_rank") <= TOP_K)
    else:
        sp = pl.read_parquet(NORM / "cand" / f"{name}_sparse" / "*.parquet")
        de = pl.read_parquet(NORM / "cand" / f"{name}_dense" / "*.parquet")
    return add_extras(merge_channels(sp, de), load_extras(name))


def load_extras(name: str) -> pl.DataFrame | None:
    """Name-only extra candidates (extras.py) of the records without an address; None when not built."""
    p = NORM / "cand" / f"{name}_extra.parquet"
    return pl.read_parquet(p) if p.exists() and os.environ.get("ER_EXTRAS", "1") == "1" else None


def add_extras(c: pl.DataFrame, ex: pl.DataFrame | None) -> pl.DataFrame:
    """Append the extras as extra_rank 1..5 rows (no regular-channel scores: rank 99 / score defaults); regular rows get 0."""
    c = c.with_columns(pl.lit(0, pl.Int8).alias("extra_rank"))
    if ex is None or ex.height == 0:
        return c
    ex = ex.join(c.select("rec").unique(), on="rec", how="semi")
    rows = ex.with_columns(pl.lit(0.0).alias("sparse_score"), pl.lit(99).alias("sparse_rank"),
                           pl.lit(-1.0).alias("dense_score"), pl.lit(99).alias("dense_rank")).select(c.columns)
    return pl.concat([c, rows.cast(c.schema)])


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
    t = structfeat.entity_lists(texts)  # + numbers, unit tokens, rare/generic name words, legal / state codes
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
    c = structfeat.add_pair_features(c.with_columns([pl.Series(k, v) for k, v in feats.items()]))

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


FEATURES_BASE = ["sparse_score", "sparse_rank", "dense_score", "dense_rank", "both", "dense_gap_best", "sparse_gap_best",
                 "n_cands", "dense_margin", "name_ratio", "name_tset", "name_tsort", "name_partial",
                 "name_jw", "name_native_ratio", "addr_tset", "addr_partial", "city_eq", "state_eq", "postal_eq",
                 "legal_form_eq", "num_eq", "any_domain", "q_has_addr", "s_has_addr", "q_name_len", "s_name_len", "from_s3",
                 "extra_rank"]
# ER_STRUCT=0 -> the previous 29-feature model (ablation); ER_STATE_CATS=0 -> keep the pair verdict but not the raw state codes
_STRUCT = [f for f in structfeat.STRUCT_FEATURES
           if os.environ.get("ER_STATE_CATS", "1") == "1" or f not in ("state_code_q", "state_code_s")]
FEATURES = FEATURES_BASE + (_STRUCT if os.environ.get("ER_STRUCT", "1") == "1" else [])
# optional cross-encoder evidence (crossenc.py): NaN where a pair was not scored
if os.environ.get("ER_CE", "0") == "1":
    FEATURES = FEATURES + ["ce_score", "ce_gap_best"] + (["ce2_score", "ce2_gap_best"] if CE2_TAG else [])
CATEGORICAL = [f for f in structfeat.CATEGORICAL if f in FEATURES]


def attach_ce(f: pl.DataFrame, name: str) -> pl.DataFrame:
    """ce_score = cross-encoder probability of a pair (null where it was not scored), ce_gap_best = best score of the record minus this one.
    With ER_CE2_TAG a second cross-encoder adds ce2_score / ce2_gap_best the same way."""
    if os.environ.get("ER_CE", "0") != "1":
        return join_ce(f, None)  # cross-encoder features are not part of this model
    for tag, col in ((CE_TAG, "ce"), (CE2_TAG, "ce2")):
        if col == "ce2" and not tag:
            break
        p = NORM / f"ce_{name}{tag}.parquet"
        if not p.exists():  # ER_CE=1 asks for them: never fall back to silent NaNs
            raise FileNotFoundError(f"ER_CE=1 but {p.name} is missing - run `python crossenc.py score {name}` first (or unset ER_CE)")
        f = join_ce(f, pl.read_parquet(p), col)
    return f


def join_ce(f: pl.DataFrame, ce: pl.DataFrame | None, col: str = "ce") -> pl.DataFrame:
    """Attach cross-encoder columns <col>_score / <col>_gap_best (all null when ce is None or the pair was not scored)."""
    if ce is None:
        return f.with_columns(pl.lit(None, pl.Float32).alias(f"{col}_score"), pl.lit(None, pl.Float32).alias(f"{col}_gap_best"))
    f = f.join(ce.select("rec", "s1", pl.col("ce_score").cast(pl.Float32).alias(f"{col}_score")), on=["rec", "s1"], how="left")
    return f.with_columns((pl.col(f"{col}_score").max().over("rec") - pl.col(f"{col}_score")).alias(f"{col}_gap_best"))


def build_features(name: str):
    t0 = time.time()
    split = "test" if name == "test" else "train"
    prefix = "" if split == "train" else "test_"
    c = retrieval_features(candidates(name))
    print(f"{name}: {c.height:,} pairs, {c['rec'].n_unique():,} records ({time.time() - t0:.0f}s)", flush=True)
    need = pl.concat([c.select(pl.col("rec").alias("entity_id")), c.select(pl.col("s1").alias("entity_id"))]).unique()
    texts = pl.concat([read_texts(prefix, i).join(need, on="entity_id", how="semi") for i in (1, 2, 3)])  # one source at a time, keep only the rows the candidates touch
    parts = []
    for i in range(0, c.height, FEAT_CHUNK):
        parts.append(string_features(c.slice(i, FEAT_CHUNK), texts))
        print(f"  features {min(i + FEAT_CHUNK, c.height):,}/{c.height:,} ({time.time() - t0:.0f}s)", flush=True)
    f = pl.concat(parts)
    f = attach_ce(f, name)
    if split == "train":
        f = f.join(ground_truth().with_columns(pl.lit(1, pl.Int8).alias("label")), on=["rec", "s1"], how="left") \
             .with_columns(pl.col("label").fill_null(0))
    f.write_parquet(NORM / f"feat_{name}{FEAT_TAG}.parquet")
    print(f"saved feat_{name} ({time.time() - t0:.0f}s)", flush=True)


def assign(f: pl.DataFrame, p: np.ndarray, thr: float) -> pl.DataFrame:
    """One owner per record: keep the record's best candidate if its probability clears the threshold."""
    return (f.select("rec", "s1").with_columns(pl.Series("p", p))
             .sort("p", descending=True).group_by("rec", maintain_order=True).head(1).filter(pl.col("p") >= thr))


# ----------------------------------------------------------------------------------------- hard-example training
HARD_POS_BELOW = 0.90   # a true pair the mining model gives < 0.90 is a hard positive
HARD_NEG_ABOVE = 0.10   # a wrong pair the mining model gives > 0.10 is a hard negative (twins, look-alikes)
HARD_WEIGHT = 3.0       # same weight for hard positives and hard negatives -> the odds inside the hard region stay calibrated
EASY_NEG_KEEP = 0.15    # easy negatives are down-sampled and re-weighted by 1/keep (faster, same prior)
MINE_ROWS = 3_000_000


def mine_hard(x: np.ndarray, y: np.ndarray, params: dict, seed: int = 0):
    """Row weights + keep mask from a deliberately weak mining model, so its in-sample scores behave like held-out ones
    (a 127-leaf model with 2000 trees would memorise the training pairs and hide every hard example)."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), size=min(MINE_ROWS, len(y)), replace=False)
    weak = dict(params, num_leaves=31, min_data_in_leaf=2000, learning_rate=0.1, feature_fraction=0.7)
    mine = lgb.train(weak, lgb.Dataset(x[sub], y[sub], feature_name=FEATURES, categorical_feature=CATEGORICAL or "auto"), 150)
    pm = mine.predict(x)
    hard_pos = (y == 1) & (pm < HARD_POS_BELOW)
    hard_neg = (y == 0) & (pm > HARD_NEG_ABOVE)
    easy_neg = (y == 0) & ~hard_neg
    keep = ~easy_neg | (rng.random(len(y)) < EASY_NEG_KEEP)
    w = np.ones(len(y), dtype=np.float32)
    w[hard_pos | hard_neg] = HARD_WEIGHT
    w[easy_neg] = 1.0 / EASY_NEG_KEEP
    print(f"hard mining: {hard_pos.sum():,} hard positives ({hard_pos.sum() / max(1, (y == 1).sum()):.1%} of positives), "
          f"{hard_neg.sum():,} hard negatives, easy negatives kept {keep[easy_neg].mean():.0%}; training rows {keep.sum():,} of {len(y):,}", flush=True)
    return w[keep], keep


# ----------------------------------------------------------------------------------------- decision rule
DECISION_PATH = ROOT / "models" / os.environ.get("ER_DECISION", "decision.json")
REAL_ORPHAN_SHARE = 1 - 7638365 / (5034616 + 5285603)   # share of S2/S3 records without an S1 owner in the training data


def record_table(ev: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """One row per record: best / second-best probability, correctness of the best pair, address flag."""
    d = ev.select("rec", "s1", "label", "q_has_addr").with_columns(pl.Series("p", p)).sort("p", descending=True)
    return d.group_by("rec", maintain_order=True).agg(
        pl.col("p").first().alias("p1"), pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
        pl.col("label").first().alias("label1"), pl.col("q_has_addr").first().alias("has_addr"))


def tune_decision(rt: pl.DataFrame, owned: pl.DataFrame, n_owned: int, orphan_w: float, label: str = "") -> dict:
    """Grid over (threshold for records WITH address, threshold for records WITHOUT, margin best-vs-second).
    Objective: F0.5 with orphan false positives re-weighted to the real orphan share of the data."""
    d = rt.join(owned.select("rec").with_columns(pl.lit(True).alias("owned")), on="rec", how="left").with_columns(pl.col("owned").fill_null(False))
    p1, p2, lab, adr, own = (d["p1"].to_numpy(), d["p2"].to_numpy(), d["label1"].to_numpy(), d["has_addr"].to_numpy().astype(bool), d["owned"].to_numpy())
    grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    best = (-1.0, None)
    for ta in grid:
        for tn in grid:
            for mg in (0.0, 0.05, 0.1, 0.2, 0.3):
                acc = (p1 >= np.where(adr, ta, tn)) & ((p1 - p2) >= mg)
                tp = (acc & (lab == 1)).sum(); wo = (acc & (lab == 0) & own).sum(); orph = (acc & ~own).sum()
                fp = wo + orphan_w * orph
                if tp == 0:
                    continue
                prec, rec = tp / (tp + fp), tp / n_owned
                f = 1.25 * prec * rec / (0.25 * prec + rec)
                if f > best[0]:
                    best = (f, dict(thr_addr=ta, thr_noaddr=tn, margin=mg, precision=float(prec), recall=float(rec), f05=float(f)))
    print(f"  decision {label}: {best[1]}", flush=True)
    return best[1]


def apply_decision(rt: pl.DataFrame, owned: pl.DataFrame, n_owned: int, orphan_w: float, dec: dict) -> dict:
    d = rt.join(owned.select("rec").with_columns(pl.lit(True).alias("owned")), on="rec", how="left").with_columns(pl.col("owned").fill_null(False))
    p1, p2, lab, adr, own = (d["p1"].to_numpy(), d["p2"].to_numpy(), d["label1"].to_numpy(), d["has_addr"].to_numpy().astype(bool), d["owned"].to_numpy())
    acc = (p1 >= np.where(adr, dec["thr_addr"], dec["thr_noaddr"])) & ((p1 - p2) >= dec["margin"])
    tp = (acc & (lab == 1)).sum(); wo = (acc & (lab == 0) & own).sum(); orph = (acc & ~own).sum()
    fp = wo + orphan_w * orph
    prec, rec = tp / (tp + fp), tp / n_owned
    return dict(tp=int(tp), fn=int(n_owned - tp), fp_wrong_owner=int(wo), fp_orphan_weighted=float(orphan_w * orph),
                precision=float(prec), recall=float(rec), f05=float(1.25 * prec * rec / (0.25 * prec + rec)))


def expected_official(tp: int, n_owned: int, fp: float) -> float:
    """Expected per-S1 macro F0.5 (singletons count) implied by pair-level error rates, S1 sizes from the training ground truth."""
    from block import DATASET
    g = pl.read_csv(DATASET / "train" / "train_ground_truth.tsv", separator="\t", infer_schema_length=0, quote_char=None)
    n = g.select(pl.when(pl.col("matched_entity_ids").is_null()).then(0)
                   .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1).alias("n"))["n"].to_numpy()
    rng = np.random.default_rng(0)
    N = 400_000
    nn = rng.choice(n, N)
    t_ = rng.binomial(nn, tp / n_owned); f_ = rng.poisson(fp / n_owned * n.mean(), N)
    pred = t_ + f_
    pr = np.where(pred > 0, t_ / np.maximum(pred, 1), 0.0); rc = np.where(nn > 0, t_ / np.maximum(nn, 1), 0.0)
    f = np.where(pr + rc > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    return float(np.where(nn == 0, np.where(pred == 0, 1.0, 0.0), np.where(pred == 0, 0.0, f)).mean())


def fit(max_rows: int | None = None, out: Path = MODEL_PATH, hard: bool = True):
    """Fit the ranker. Hard positives / hard negatives get more weight; the decision rule (thresholds per record type +
    margin) is tuned on one half of the held-out entities and reported on the other half, so the reported score is not
    the one that was optimised. The other half also drives early stopping only for the model, never for the thresholds."""
    import gc
    import lightgbm as lgb
    tr = pl.read_parquet(NORM / f"feat_train{FEAT_TAG}.parquet", n_rows=max_rows)
    ev = pl.read_parquet(NORM / f"feat_eval{FEAT_TAG}.parquet", n_rows=max_rows)
    print(f"train pairs {tr.height:,} (pos {tr['label'].sum():,})   eval pairs {ev.height:,} (pos {ev['label'].sum():,})   features {len(FEATURES)}", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=int(os.environ.get("ER_THREADS", os.cpu_count() or 8)), max_bin=255)
    mono = [-1 if f in structfeat.CONFLICT_FLAGS else 0 for f in FEATURES]   # a conflict may lower p, never raise it (also curbs overfitting)
    if any(mono):
        params.update(monotone_constraints=mono, monotone_constraints_method="basic")
    x = tr.select(FEATURES).cast(pl.Float32).to_numpy()
    y = tr["label"].to_numpy()
    del tr
    gc.collect()
    w = None
    if hard:
        w, keep = mine_hard(x, y, params)
        x, y = x[keep], y[keep]
    dtr = lgb.Dataset(x, y, weight=w, feature_name=FEATURES, categorical_feature=CATEGORICAL or "auto", params={"max_bin": 255})
    dtr.construct()
    del x
    gc.collect()
    # held-out entities split by record: `es` early-stops the model, `thr` tunes and reports the decision rule
    es_mask = (ev["rec"].hash(seed=3) % 2 == 0).to_numpy()
    xe = ev.select(FEATURES).cast(pl.Float32).to_numpy()
    des = lgb.Dataset(xe[es_mask], ev["label"].to_numpy()[es_mask], reference=dtr)
    m = lgb.train(params, dtr, int(os.environ.get("ER_MAX_ROUNDS", 2000)), valid_sets=[des], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(str(out))
    p = m.predict(xe)

    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    gt = ground_truth()
    owned = gt.filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi")
    n_all = qid.height
    orphan_w = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((n_all - owned.height) / owned.height)
    print(f"blocking recall on eval: {ev.filter(pl.col('label') == 1).height / owned.height:.4f}   orphan weight {orphan_w:.3f}", flush=True)
    rt = record_table(ev, p)
    rt = rt.with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("is_es"))
    half = lambda d, es: d.filter(pl.col("is_es") == es)
    res = {}
    for name, es in (("thr-half", False), ("early-stop half", True)):
        own_h = owned.join(half(rt, es).select("rec"), on="rec", how="semi")
        # owned records with no candidate at all are misses too: count them through n_owned
        n_own = owned.filter((pl.col("rec").hash(seed=3) % 2 == 0) == es).height
        res[name] = (half(rt, es), own_h, n_own)
    dec = tune_decision(*res["thr-half"][0:1], res["thr-half"][1], res["thr-half"][2], orphan_w, "tuned on thr-half")
    held = {}
    for name, (r_, o_, n_) in res.items():
        r = apply_decision(r_, o_, n_, orphan_w, dec)
        off = expected_official(r["tp"], n_, r["fp_wrong_owner"] + r["fp_orphan_weighted"])
        held[name] = dict(r, official=off)
        print(f"  {name:16s} TP {r['tp']:,}  FN {r['fn']:,}  FP wrong-owner {r['fp_wrong_owner']}  FP orphan (reweighted) {r['fp_orphan_weighted']:.0f}   "
              f"precision {r['precision']:.4f}  recall {r['recall']:.4f}  F0.5 {r['f05']:.4f}   expected official score {off:.4f}", flush=True)
    DECISION_PATH.parent.mkdir(exist_ok=True)
    DECISION_PATH.write_text(json.dumps(dict(dec, held=held)), encoding="utf-8")  # `held`: what main.py compares between models
    imp = sorted(zip(FEATURES, m.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", ", ".join(k for k, _ in imp[:14]), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "features":
        build_features(sys.argv[2])
    elif sys.argv[1] == "fit":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None  # a row limit means smoke test: never overwrite the real model
        fit(n, MODEL_PATH if n is None else MODEL_PATH.with_name("ranker_smoketest.txt"), hard=os.environ.get("ER_HARD", "1") == "1")
