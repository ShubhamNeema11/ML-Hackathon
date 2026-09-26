"""Separate pipeline for records WITHOUT an address (about 65% of the remaining errors, from ~4% of the owned records).

Two pipelines: records with an address keep the general ranker (ranker.py, ER_ADDR_ONLY=1 fits it on address records only) and the
cross-encoder (ER_CE_ADDR_ONLY=1 scores only those); records without an address get this specialist and NO cross-encoder (it added
+0.002 F0.5 on the group, within noise). Nothing expensive is retrained.

Candidates: the regular top-10 sparse U top-10 dense, plus the char 3-gram TF-IDF list of blocking_noaddr.py (top-10 + near-ties, cap 100).
Model: a small LightGBM on the ~31k no-address training records; features add tf_* (score / rank / gap / tie-cluster size of the TF-IDF
list) and what a name-only record needs: candidates sharing (almost) the same name, name gap to the record's best, S1 name twins.

  python blocking_noaddr.py build train|eval|test         TF-IDF candidates -> normalized/cand/na_tfidf_<split>.parquet
  python noaddr.py features train|eval|test               -> normalized/feat_noaddr2_<split>.parquet   (ER_NA_TFIDF=0: the old 30-extras variant)
  python noaddr.py fit                                    -> models/noaddr_c.txt, comparison with B2 on the no-address group
  ER_FEAT_TAG=_ce2 ER_B2=ranker_addr.txt python noaddr.py joint            -> whole held-out set: address model + specialist, decision -> models/decision_na.json
  python noaddr.py apply                                  -> patches a finished score dir (pred_final -> pred_final_na) for records without address
  ER_CE=1 ER_CE_TAG=_v2 ER_ADDR_ONLY=1 ER_MODEL=ranker_addr.txt ER_DECISION=decision_addr.json python ranker.py fit     address-only LightGBM
"""
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

import ranker
from ranker import FEATURES, NORM, ground_truth, read_texts, retrieval_features, string_features

TFIDF = os.environ.get("ER_NA_TFIDF", "1") == "1"   # candidates from blocking_noaddr.py (char 3-gram TF-IDF) instead of the old name-only extras
TAGV = "2" if TFIDF else ""                          # feature files feat_noaddr2_<split>.parquet (old: feat_noaddr_<split>.parquet)
MODEL = ranker.ROOT / "models" / os.environ.get("ER_NOADDR_MODEL", "noaddr_c.txt" if TFIDF else "noaddr_a.txt")
NEW = ["name_ratio_gap", "name_tset_gap", "name_jw_gap", "name_ratio_rank", "n_name97", "n_name90", "s_name_twins", "cand_dup_frac"]
TF = ["tf_score", "tf_rank", "tf_gap_best", "tf_ratio", "tf_ties", "tf_n_close"]
COLS = FEATURES + NEW + (TF if TFIDF else [])
A_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=5.0, verbose=-1, num_threads=os.cpu_count() or 8)


def trainall_queries():
    """Every no-address training record the specialist may learn from: not a held-out evaluation query and not owned by a held-out S1 entity
    (the same rule as block.ranker_train_queries, without the 10% sample) -> normalized/trainall_queries.parquet."""
    from block import is_val_s1
    q = (pl.concat([pl.scan_parquet(NORM / f"source{i}.parquet").select("entity_id", "has_addr") for i in (2, 3)])
           .filter(~pl.col("has_addr")).select("entity_id").collect())
    owner = ground_truth().rename({"rec": "entity_id"})
    q = q.join(owner, on="entity_id", how="left")
    # owned by a held-out S1 -> excluded; orphans in pass 2's held-out orphan sample (pass2.populations: hash(seed=7) % 10 == 0) -> excluded too
    q = q.filter((pl.col("s1").is_not_null() & ~is_val_s1()) | (pl.col("s1").is_null() & (pl.col("entity_id").hash(seed=7) % 10 != 0))).select("entity_id")
    q = q.join(pl.read_parquet(NORM / "eval_queries.parquet"), on="entity_id", how="anti")
    q.write_parquet(NORM / "trainall_queries.parquet")
    print(f"trainall: {q.height:,} no-address training records (held-out entities and evaluation queries excluded)", flush=True)


def regular_sparse():
    """The regular sparse channel (block.py keys, top-10) for the trainall records -> normalized/cand/trainall_sparse/part0000.parquet."""
    from block import SparseIndex, load
    q = pl.concat([load("train", i) for i in (2, 3)]).join(pl.read_parquet(NORM / "trainall_queries.parquet"), on="entity_id", how="semi")
    SparseIndex(load("train", 1)).search(q, out_dir=NORM / "cand" / "trainall_sparse")
    print(f"trainall sparse candidates for {q.height:,} records", flush=True)


def regular_dense():
    """The regular dense channel (the fine-tuned e5, top-10) for the trainall records -> normalized/cand/trainall_dense/ (GPU, own process)."""
    import embed
    keep = pl.read_parquet(NORM / "trainall_queries.parquet")
    qt = pl.concat([embed.texts("train", i).join(keep, on="entity_id", how="semi") for i in (2, 3)])
    embed.search(embed.texts("train", 1), lambda c: [qt.filter(pl.col("country") == c)], embed.load_model(), out_dir=NORM / "cand" / "trainall_dense")
    print(f"trainall dense candidates for {qt.height:,} records", flush=True)


def s1_name_counts(prefix: str) -> pl.DataFrame:
    """How many S1 entities carry each exact (country, name_core): a candidate whose name is shared is a poorer bet."""
    d = pl.read_parquet(NORM / f"{prefix}source1.parquet", columns=["entity_id", "country", "name_core"])
    c = d.group_by("country", "name_core").agg(pl.len().alias("s_name_twins"))
    return d.join(c, on=["country", "name_core"]).select(pl.col("entity_id").alias("s1"), "s_name_twins")


def relative_features(f: pl.DataFrame, twins: pl.DataFrame) -> pl.DataFrame:
    g = "rec"
    return (f.join(twins, on="s1", how="left")
             .with_columns((pl.col("name_ratio").max().over(g) - pl.col("name_ratio")).alias("name_ratio_gap"),
                           (pl.col("name_tset").max().over(g) - pl.col("name_tset")).alias("name_tset_gap"),
                           (pl.col("name_jw").max().over(g) - pl.col("name_jw")).alias("name_jw_gap"),
                           pl.col("name_ratio").rank("min", descending=True).over(g).alias("name_ratio_rank"),
                           (pl.col("name_ratio") >= 97).sum().over(g).alias("n_name97"),
                           (pl.col("name_ratio") >= 90).sum().over(g).alias("n_name90"),
                           pl.col("s_name_twins").fill_null(0))
             .with_columns((pl.col("n_name90") / pl.col("n_cands")).alias("cand_dup_frac")))


def add_tfidf(c: pl.DataFrame, split: str, lim: int) -> pl.DataFrame:
    """Union of the regular candidates and the TF-IDF list (blocking_noaddr.py); TF-IDF-only pairs get the 'no regular score' defaults of
    ranker.add_extras plus extra_rank 1. tf_* features: score, rank, gap to the record's best, ratio to it, size of the tie cluster, number
    of near-best candidates (twin clusters are what a name-only ranker cannot separate)."""
    t = pl.read_parquet(NORM / "cand" / f"na_tfidf_{split}{'_smoke' if lim else ''}.parquet")
    t = t.join(t.group_by("rec").agg(pl.col("score").max().alias("best")), on="rec")
    t = t.with_columns((pl.col("best") - pl.col("score")).alias("tf_gap_best"), (pl.col("score") / pl.col("best")).alias("tf_ratio"),
                       (pl.col("score") >= 0.85 * pl.col("best")).sum().over("rec").alias("tf_n_close"),
                       pl.len().over("rec", pl.col("score").round(5)).alias("tf_ties"))
    t = t.select("rec", "s1", pl.col("score").cast(pl.Float32).alias("tf_score"), pl.col("rank").cast(pl.Int16).alias("tf_rank"),
                 pl.col("tf_gap_best").cast(pl.Float32), pl.col("tf_ratio").cast(pl.Float32), pl.col("tf_n_close").cast(pl.Int32), pl.col("tf_ties").cast(pl.Int32))
    only = t.join(c.select("rec", "s1"), on=["rec", "s1"], how="anti")
    rows = only.select("rec", "s1").with_columns(pl.lit(0.0).alias("sparse_score"), pl.lit(99).alias("sparse_rank"), pl.lit(-1.0).alias("dense_score"),
                                                  pl.lit(99).alias("dense_rank"), pl.lit(1, pl.Int8).alias("extra_rank")).select(c.columns)
    c = pl.concat([c, rows.cast(c.schema)])
    return c.join(t.drop("rec", "s1") if False else t, on=["rec", "s1"], how="left")


def build(split: str):
    """Features of all candidates (regular top-10 U top-10 plus the deep name-only list) of the records without an address."""
    prefix = "" if split != "test" else "test_"
    ids = pl.concat([pl.scan_parquet(NORM / f"{prefix}source{i}.parquet").select("entity_id", "has_addr") for i in (2, 3)]).filter(~pl.col("has_addr")).select("entity_id").collect()
    q = {"train": "train_queries", "eval": "eval_queries", "trainall": "trainall_queries"}.get(split)
    if q:
        ids = ids.join(pl.read_parquet(NORM / f"{q}.parquet"), on="entity_id", how="semi")
    rec = ids.rename({"entity_id": "rec"})
    lim = int(os.environ.get("ER_NOADDR_LIMIT", 0))  # smoke test: only the first N records
    if lim:
        rec = rec.head(lim)
    extras = None if TFIDF else True
    if split == "test":  # 180M test pairs do not fit in memory: keep only these records' rows, part by part
        keep = lambda d: d.join(rec, on="rec", how="semi")
        sp = pl.concat([keep(pl.read_parquet(p)) for p in sorted((NORM / "cand" / "test_sparse").glob("part*.parquet"))])
        de = pl.concat([keep(pl.read_parquet(p)) for p in sorted((NORM / "cand" / "test_dense").glob("*.parquet"))])
        c = ranker.merge_channels(sp, de)
        c = ranker.add_extras(c, None if TFIDF else ranker.load_extras("test")).join(rec, on="rec", how="semi")
    else:
        if TFIDF:
            os.environ["ER_EXTRAS"] = "0"   # the regular top-10 U top-10 only; the name-only part comes from the TF-IDF list
        c = ranker.candidates(split).join(rec, on="rec", how="semi")
    if TFIDF:
        c = add_tfidf(c, split, lim)
    c = retrieval_features(c)
    need = pl.concat([c.select(pl.col("rec").alias("entity_id")), c.select(pl.col("s1").alias("entity_id"))]).unique()
    texts = pl.concat([read_texts(prefix, i).join(need, on="entity_id", how="semi") for i in (1, 2, 3)])
    parts = [string_features(c.slice(i, 500_000), texts) for i in range(0, c.height, 500_000)]
    f = relative_features(pl.concat(parts), s1_name_counts(prefix))
    if split != "test":
        f = f.join(ground_truth().with_columns(pl.lit(1, pl.Int8).alias("label")), on=["rec", "s1"], how="left").with_columns(pl.col("label").fill_null(0))
    f.write_parquet(NORM / f"feat_noaddr{TAGV}_{split}{'_smoke' if lim else ''}.parquet")
    print(f"{split}: {f.height:,} pairs, {f['rec'].n_unique():,} records without address", flush=True)


CE_K = int(os.environ.get("ER_NOADDR_CE_K", 8))   # pairs per record that the existing cross-encoder scores (inference only)


def stage_a(split: str):
    """Stage-A probabilities (no CE) of every candidate, out-of-fold for train, and the top-CE_K pairs per record -> normalized/na_cepairs_<split>.parquet."""
    import lightgbm as lgb
    f = pl.read_parquet(NORM / f"feat_noaddr{TAGV}_{split}.parquet")
    x = f.select(COLS).cast(pl.Float32).to_numpy()
    if split == "train":
        y, fold = f["label"].to_numpy(), (f["rec"].hash(seed=13) % 5).to_numpy()
        p = np.zeros(len(y))
        for k in range(5):
            m = lgb.train(A_PARAMS, lgb.Dataset(x[fold != k], y[fold != k], feature_name=COLS), 900)
            p[fold == k] = m.predict(x[fold == k])
            print(f"  fold {k} done", flush=True)
    else:
        p = lgb.Booster(model_file=str(MODEL)).predict(x)
    d = f.select("rec", "s1").with_columns(pl.Series("p_a", p, dtype=pl.Float32))
    d.write_parquet(NORM / f"na_pa_{split}.parquet")
    top = d.sort("p_a", descending=True).group_by("rec", maintain_order=True).head(CE_K).select("rec", "s1")
    top.write_parquet(NORM / f"na_cepairs_{split}.parquet")
    print(f"{split}: stage A done, {top.height:,} pairs for the cross-encoder", flush=True)


def with_ce(split: str) -> pl.DataFrame:
    """Feature table + p_a + cross-encoder score/gap (NaN where a pair was not among the record's top-CE_K of stage A)."""
    f = pl.read_parquet(NORM / f"feat_noaddr{TAGV}_{split}.parquet").join(pl.read_parquet(NORM / f"na_pa_{split}.parquet"), on=["rec", "s1"], how="left")
    ce = pl.read_parquet(NORM / f"ce_{split}_na.parquet")
    return ranker.join_ce(f, ce, "ce")

COLS_B = COLS + ["p_a", "ce_score", "ce_gap_best"]
MODEL_B = ranker.ROOT / "models" / os.environ.get("ER_NOADDR_MODEL_B", "noaddr_b.txt")


def fit_b():
    """Stage B: stage-A probability + cross-encoder score of the record's top-CE_K pairs on top of the specialist features."""
    import lightgbm as lgb
    tr, ev = with_ce("train"), with_ce("eval")
    x, y = tr.select(COLS_B).cast(pl.Float32).to_numpy(), tr["label"].to_numpy()
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi").join(ev.select("rec").unique(), on="rec", how="semi")
    owned_recs, n_owned = set(owned["rec"].to_list()), owned.height
    es = (tr["rec"].hash(seed=11) % 5 == 0).to_numpy()
    m = lgb.train(A_PARAMS, lgb.Dataset(x[~es], y[~es], feature_name=COLS_B), 3000, valid_sets=[lgb.Dataset(x[es], y[es])],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(400)])
    m.save_model(str(MODEL_B))
    p_b = m.predict(ev.select(COLS_B).cast(pl.Float32).to_numpy())
    for w in (0.1,):
        print(f"orphan weight {w}:", flush=True)
        group_score(record_rt(ev, ev["p_a"].to_numpy()), owned_recs, n_owned, w, "stage A (no CE)")
        group_score(record_rt(ev, p_b), owned_recs, n_owned, w, "stage B (A + CE top-%d)" % CE_K)
    imp = sorted(zip(COLS_B, m.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", ", ".join(k for k, _ in imp[:10]), flush=True)

# ------------------------------------------------------------------------------------------------ joint decision
# Orphan weight: eval queries hold every sampled orphan but only the held-out 10% of owned records, so orphans are over-represented by ~10x; the
# same factor applies to records without an address (ground truth: only 739 of 31,031 no-address training records are orphans, ~2.4%),
# so ONE weight (ranker's global orphan_w, ~0.1) prices both groups. (An earlier version used 0.76 for no-address records: wrong, it
# counted records whose owner blocking had missed as orphans.)


def tune_joint(rt: pl.DataFrame, own: np.ndarray, w: np.ndarray, n_owned: int):
    """ranker.tune_decision with a per-record orphan weight, so both groups are priced with their own orphan share."""
    p1, p2, lab, adr = rt["p1"].to_numpy(), rt["p2"].to_numpy(), rt["label1"].to_numpy(), rt["has_addr"].to_numpy().astype(bool)
    grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    best = (-1.0, None)
    for ta in grid:
        for tn in grid:
            for mg in (0.0, 0.05, 0.1, 0.2, 0.3):
                for mn in (0.0, 0.1, 0.2, 0.3, 0.4):
                    acc = (p1 >= np.where(adr, ta, tn)) & ((p1 - p2) >= np.where(adr, mg, mn))
                    tp = int((acc & (lab == 1)).sum())
                    fp = float((acc & (lab == 0) & own).sum() + (w * (acc & ~own)).sum())
                    if tp == 0:
                        continue
                    pr, rc = tp / (tp + fp), tp / n_owned
                    f = 1.25 * pr * rc / (0.25 * pr + rc)
                    if f > best[0]:
                        best = (f, dict(thr_addr=ta, thr_noaddr=tn, margin=mg, margin_noaddr=mn))
    return best[1]


def apply_joint(rt, own, w, n_owned, dec):
    p1, p2, lab, adr = rt["p1"].to_numpy(), rt["p2"].to_numpy(), rt["label1"].to_numpy(), rt["has_addr"].to_numpy().astype(bool)
    acc = (p1 >= np.where(adr, dec["thr_addr"], dec["thr_noaddr"])) & ((p1 - p2) >= np.where(adr, dec["margin"], dec.get("margin_noaddr", dec["margin"])))
    tp = int((acc & (lab == 1)).sum())
    fp_wo = int((acc & (lab == 0) & own).sum())
    fp_or = float((w * (acc & ~own)).sum())
    pr, rc = tp / (tp + fp_wo + fp_or), tp / n_owned
    return dict(tp=tp, fn=int(n_owned - tp), fp_wrong_owner=fp_wo, fp_orphan_weighted=fp_or, precision=float(pr), recall=float(rc),
                f05=float(1.25 * pr * rc / (0.25 * pr + rc)),
                official=ranker.expected_official(tp, n_owned, fp_wo + fp_or))


def joint():
    """Whole held-out set: B2 for records with an address, the specialist for records without; same two-half protocol as ranker.fit
    (thresholds tuned on the 'thr' half, reported on the other), compared with B2 for everybody under identical pricing."""
    import lightgbm as lgb
    b2 = lgb.Booster(model_file=str(ranker.ROOT / "models" / os.environ.get("ER_B2", "ranker_b2.txt")))
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet")
    ev = ev.with_columns(pl.Series("p", b2.predict(ev.select(b2.feature_name()).cast(pl.Float32).to_numpy())))
    na = pl.read_parquet(NORM / f"feat_noaddr{TAGV}_eval.parquet")
    sp = lgb.Booster(model_file=str(MODEL))
    na = na.with_columns(pl.Series("p", sp.predict(na.select(COLS).cast(pl.Float32).to_numpy())))
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi")
    n_all, orphan_w = qid.height, None
    orphan_w = (ranker.REAL_ORPHAN_SHARE / (1 - ranker.REAL_ORPHAN_SHARE)) / ((n_all - owned.height) / owned.height)
    w_na = orphan_w
    print(f"orphan weights: addr {orphan_w:.3f}  no-addr {w_na:.3f}", flush=True)
    owned_set = set(owned["rec"].to_list())
    half = lambda d: (d["rec"].hash(seed=3) % 2 == 0)

    def table(parts_p):
        rt = pl.concat([record_table(e, p) for e, p in parts_p]).with_columns(half(pl.DataFrame()) if False else (pl.col("rec").hash(seed=3) % 2 == 0).alias("is_es"))
        own = np.array([r in owned_set for r in rt["rec"].to_list()])
        w = np.where(rt["has_addr"].to_numpy().astype(bool), orphan_w, w_na)
        return rt, own, w

    record_table = ranker.record_table
    ev_addr = ev.filter(pl.col("q_has_addr") == 1)
    ev_na_old = ev.filter(pl.col("q_has_addr") == 0)
    variants = {"B2 for everyone": [(ev_addr, ev_addr["p"].to_numpy()), (ev_na_old, ev_na_old["p"].to_numpy())],
                "B2 + no-address specialist": [(ev_addr, ev_addr["p"].to_numpy()), (na.with_columns(pl.lit(0).cast(pl.Int8).alias("q_has_addr")), na["p"].to_numpy())]}
    res = {}
    for name, parts in variants.items():
        rt, own, w = table(parts)
        es = rt["is_es"].to_numpy()
        n_half = {k: owned.filter((pl.col("rec").hash(seed=3) % 2 == 0) == k).height for k in (False, True)}
        dec = tune_joint(rt.filter(~pl.Series(es)), own[~es], w[~es], n_half[False])
        held = {"thr-half": apply_joint(rt.filter(~pl.Series(es)), own[~es], w[~es], n_half[False], dec),
                "early-stop half": apply_joint(rt.filter(pl.Series(es)), own[es], w[es], n_half[True], dec)}
        res[name] = dict(dec, held=held)
        print(f"RESULT {name:28s} {dec}  thr-half {held['thr-half']['official']:.4f}   report-half {held['early-stop half']['official']:.4f}"
              f"   (TP {held['early-stop half']['tp']}, FP wrong {held['early-stop half']['fp_wrong_owner']}, FP orphan-w {held['early-stop half']['fp_orphan_weighted']:.0f})", flush=True)
    import json
    out = ranker.ROOT / "models" / os.environ.get("ER_DECISION", "decision_na.json")
    out.write_text(json.dumps(dict(res["B2 + no-address specialist"], baseline_held=res["B2 for everyone"]["held"], baseline_decision={k: v for k, v in res["B2 for everyone"].items() if k != "held"})), encoding="utf-8")
    print("wrote", out.name, flush=True)

def apply():
    """Patch a finished pass-1 score directory: the rows of records without an address are replaced by the specialist's scores.
    ER_PRED_IN (default normalized/pred_final) -> ER_PRED_OUT (default normalized/pred_final_na). Everything else is copied unchanged, so
    predict.py write / pass2.py read the result like any other score directory (use decision_na.json)."""
    import shutil
    import lightgbm as lgb
    from predict import ids_table
    pin = Path(os.environ.get("ER_PRED_IN", NORM / "pred_final"))
    pout = Path(os.environ.get("ER_PRED_OUT", NORM / "pred_final_na"))
    tag = "_smoke" if os.environ.get("ER_NOADDR_LIMIT") else ""
    f = pl.read_parquet(NORM / f"feat_noaddr{TAGV}_test{tag}.parquet")
    p = lgb.Booster(model_file=str(MODEL)).predict(f.select(COLS).cast(pl.Float32).to_numpy())
    ids = ids_table()
    rows = (f.select("rec", "s1").with_columns(pl.Series("p", p, dtype=pl.Float32))
             .join(ids.rename({"entity_id": "rec", "idx": "rec_i"}), on="rec").join(ids.rename({"entity_id": "s1", "idx": "s1_i"}), on="s1")
             .select("rec_i", "s1_i", "p"))
    done = rows.select("rec_i").unique()
    pout.mkdir(parents=True, exist_ok=True)
    n_out = 0
    for q in sorted(pin.glob("part*.parquet")):
        d = pl.read_parquet(q)
        kept = d.join(done.cast({"rec_i": d.schema["rec_i"]}), on="rec_i", how="anti")
        n_out += d.height - kept.height
        kept.write_parquet(pout / q.name)
    rows = rows.cast({"rec_i": d.schema["rec_i"], "s1_i": d.schema["s1_i"]})
    rows.write_parquet(pout / "part9999_noaddr.parquet")
    (pout / "_DONE").touch()
    print(f"{done.height:,} records without address re-scored: {n_out:,} old rows replaced by {rows.height:,} specialist rows -> {pout}", flush=True)

def group_score(rt: pl.DataFrame, owned_recs: set, n_owned: int, orphan_w: float, label: str):
    """Best F0.5 over (threshold, margin) on the no-address group, plus what it accepts. rt: rec, p1, p2, label1."""
    p1, p2, lab = rt["p1"].to_numpy(), rt["p2"].to_numpy(), rt["label1"].to_numpy()
    own = np.array([r in owned_recs for r in rt["rec"].to_list()])
    best = None
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        for mg in (0.0, 0.1, 0.2, 0.3):
            acc = (p1 >= t) & ((p1 - p2) >= mg)
            tp = int((acc & (lab == 1)).sum())
            fp = float((acc & (lab == 0) & own).sum() + orphan_w * (acc & ~own).sum())
            if tp == 0:
                continue
            pr, rc = tp / (tp + fp), tp / n_owned
            f = 1.25 * pr * rc / (0.25 * pr + rc)
            if best is None or f > best[0]:
                best = (f, t, mg, tp, fp, pr, rc)
    print(f"  {label:28s} F0.5 {best[0]:.4f}  thr {best[1]} margin {best[2]}  TP {best[3]}  FP {best[4]:.1f}  precision {best[5]:.4f}  recall {best[6]:.4f}", flush=True)
    return best


def record_rt(f: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    d = f.select("rec", "label").with_columns(pl.Series("p", p)).sort("p", descending=True)
    return d.group_by("rec", maintain_order=True).agg(pl.col("p").first().alias("p1"), pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
                                                      pl.col("label").first().alias("label1"))


def fit():
    import lightgbm as lgb
    tr_split = "trainall" if (NORM / f"feat_noaddr{TAGV}_trainall.parquet").exists() else "train"   # all no-address training records when built
    import gc
    # only the columns the model uses (the ~10M-row trainall table would not fit next to its numpy copy otherwise)
    tr = pl.read_parquet(NORM / f"feat_noaddr{TAGV}_{tr_split}.parquet", columns=COLS + ["label", "rec"])
    ev = pl.read_parquet(NORM / f"feat_noaddr{TAGV}_eval.parquet")
    print(f"training set: feat_noaddr{TAGV}_{tr_split}.parquet", flush=True)
    # early stopping on a record-split of the TRAINING records, so the eval records stay untouched
    es = pl.Series((tr["rec"].hash(seed=11) % 5 == 0).to_numpy())
    print(f"train pairs {tr.height:,} (pos {int(tr['label'].sum()):,}) records {tr['rec'].n_unique():,}; eval pairs {ev.height:,} records {ev['rec'].n_unique():,}", flush=True)
    parts = {k: tr.filter(es if k else ~es) for k in (False, True)}
    del tr
    gc.collect()
    data = {}
    for k, d in parts.items():
        data[k] = (d.select(COLS).cast(pl.Float32).to_numpy(), d["label"].to_numpy())
        parts[k] = None
        gc.collect()
    dtr = lgb.Dataset(data[False][0], data[False][1], feature_name=COLS, free_raw_data=True)
    dtr.construct()
    des = lgb.Dataset(data[True][0], data[True][1], reference=dtr, free_raw_data=True)
    des.construct()
    del data
    gc.collect()
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi").join(ev.select("rec").unique(), on="rec", how="semi")
    owned_recs, n_owned = set(owned["rec"].to_list()), owned.height
    n_orph = ev["rec"].n_unique() - n_owned
    print(f"eval no-address: {n_owned} owned, {n_orph} orphans", flush=True)
    xe = ev.select(COLS).cast(pl.Float32).to_numpy()
    # hyperparameters (defaults = A_PARAMS, 3000 rounds): ER_NA_LR, ER_NA_LEAVES, ER_NA_MINLEAF, ER_NA_ROUNDS; chosen on the training-internal split only
    params = dict(A_PARAMS, learning_rate=float(os.environ.get("ER_NA_LR", A_PARAMS["learning_rate"])),
                  num_leaves=int(os.environ.get("ER_NA_LEAVES", A_PARAMS["num_leaves"])), min_data_in_leaf=int(os.environ.get("ER_NA_MINLEAF", A_PARAMS["min_data_in_leaf"])))
    m = lgb.train(params, dtr, int(os.environ.get("ER_NA_ROUNDS", 3000)), valid_sets=[dtr, des], valid_names=["train", "es"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    tr_ll, es_ll = m.best_score["train"]["binary_logloss"], m.best_score["es"]["binary_logloss"]
    print(f"TUNE lr {params['learning_rate']} leaves {params['num_leaves']} min_leaf {params['min_data_in_leaf']}: best iteration {m.best_iteration}, "
          f"logloss train {tr_ll:.5f} / early-stop split {es_ll:.5f} (gap {es_ll - tr_ll:.5f})", flush=True)
    m.save_model(str(MODEL))
    p_new = m.predict(xe)
    # B2 on the same eval pairs where it has them (its features come from the old files; missing pairs get 0)
    b2 = lgb.Booster(model_file=str(ranker.ROOT / "models" / os.environ.get("ER_B2", "ranker_b2.txt")))
    old = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet").filter(pl.col("q_has_addr") == 0)
    old = old.with_columns(pl.Series("p", b2.predict(old.select(b2.feature_name()).cast(pl.Float32).to_numpy()))).select("rec", "s1", "p")
    evp = ev.select("rec", "s1", "label").join(old, on=["rec", "s1"], how="left").with_columns(pl.col("p").fill_null(0.0))
    for w in (0.1,):
        print(f"orphan weight {w}:", flush=True)
        group_score(record_rt(evp, evp["p"].to_numpy()), owned_recs, n_owned, w, "B2 (current, 5 extras)")
        group_score(record_rt(ev, p_new), owned_recs, n_owned, w, "specialist (new cands, no CE)")
    imp = sorted(zip(COLS, m.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", ", ".join(k for k, _ in imp[:12]), flush=True)


if __name__ == "__main__":
    {"features": lambda: build(sys.argv[2]), "fit": fit, "stage_a": lambda: stage_a(sys.argv[2]), "fit_b": fit_b, "joint": joint, "apply": apply, "trainall_queries": trainall_queries, "regular_sparse": regular_sparse, "regular_dense": regular_dense}[sys.argv[1]]()
