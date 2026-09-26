"""Record-level accept model (experiment): instead of 'accept the best candidate if p1 >= thr and p1 - p2 >= margin', a small LightGBM decides
from the whole record: p1 / p2 / p3, the best pair's own evidence (name / address / cross-encoder / dense features), the candidate count.

Clean protocol on the held-out records: the model AND its acceptance threshold are fitted on the 'thr' half only (out-of-fold inside it),
then applied once to the report half, which nothing was fitted on. Compared with the current threshold rule on the same report half.
Address records only (records without an address are decided by the specialist, unchanged).

  ER_CE=1 ER_CE_TAG=_v2 ER_FEAT_TAG=_ce2 python accept_model.py
"""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

import noaddr
import ranker
from ranker import NORM, REAL_ORPHAN_SHARE, ground_truth

TOPF = ["name_ratio", "name_tset", "name_jw", "name_native_ratio", "addr_tset", "addr_partial", "ce_score", "ce_gap_best", "dense_score",
        "dense_margin", "sparse_score", "both", "num_equal", "num_conflict", "unit_conflict", "legal_conflict", "rare_only_q", "rare_only_s",
        "postal_eq", "city_eq", "state_eq", "from_s3", "n_cands", "q_name_len", "s_name_len"]


def record_features(ev: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    d = ev.with_columns(pl.Series("p", p)).sort("p", descending=True)
    top = d.group_by("rec", maintain_order=True).agg(
        pl.col("p").first().alias("p1"), pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
        pl.col("p").get(2, null_on_oob=True).fill_null(0.0).alias("p3"), (pl.col("p") > 0.05).sum().alias("n_plaus"),
        pl.col("label").first().alias("label1"), *[pl.col(c).first().alias(f"t_{c}") for c in TOPF])
    return top.with_columns((pl.col("p1") - pl.col("p2")).alias("m12"), pl.lit(1).alias("has_addr"))


def main():
    b2 = lgb.Booster(model_file=str(ranker.ROOT / "models" / "ranker_b2.txt"))
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet").filter(pl.col("q_has_addr") == 1)
    rt = record_features(ev, b2.predict(ev.select(b2.feature_name()).cast(pl.Float32).to_numpy()))
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi")
    orphan_w = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    rt = rt.with_columns(pl.col("rec").is_in(owned["rec"].implode()).alias("own"), (pl.col("rec").hash(seed=3) % 2 == 0).alias("is_es"))
    # "accepting is right" = the best candidate is the true owner; accepting an orphan / a wrong owner is a false positive
    y = (rt["label1"] == 1).to_numpy().astype(int)
    own, es = rt["own"].to_numpy(), rt["is_es"].to_numpy()
    w = np.where(own, 1.0, orphan_w)   # orphans are ~10x over-represented in the held-out sample: weight them to the real share
    feats = ["p1", "p2", "p3", "m12", "n_plaus"] + [f"t_{c}" for c in TOPF]
    X = rt.select(feats).cast(pl.Float32).to_numpy()
    n_owned = {k: owned.filter((pl.col("rec").hash(seed=3) % 2 == 0) == k).height for k in (False, True)}
    # the owned records WITHOUT an address count in the denominators but are not decided here: evaluate address records only,
    # with recall relative to the owned address records of each half
    owned_addr = owned.join(rt.select("rec"), on="rec", how="semi")
    n_own_addr = {k: owned_addr.filter((pl.col("rec").hash(seed=3) % 2 == 0) == k).height for k in (False, True)}
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbose=-1)

    def score(acc, mask, n_own):
        tp = (acc & (y == 1) & mask).sum(); wo = (acc & (y == 0) & own & mask).sum(); orph = (acc & ~own & mask).sum()
        fp = wo + orphan_w * orph
        pr, rc = tp / (tp + fp), tp / n_own
        return dict(tp=int(tp), fp_wrong=int(wo), fp_orph_w=float(orphan_w * orph), f05=float(1.25 * pr * rc / (0.25 * pr + rc)),
                    official=ranker.expected_official(int(tp), n_own, float(fp)))

    A, B = ~es, es
    # out-of-fold q on the thr half (5 folds by record) -> choose the acceptance threshold there
    fold = (rt["rec"].hash(seed=17) % 5).to_numpy()
    q = np.zeros(len(y))
    for k in range(5):
        tr, te = A & (fold != k), A & (fold == k)
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr]), 400)
        q[te] = m.predict(X[te])
    best = max(((t, score(q >= t, A, n_own_addr[False])["official"]) for t in np.arange(0.30, 0.99, 0.01)), key=lambda z: z[1])
    m = lgb.train(params, lgb.Dataset(X[A], y[A], weight=w[A]), 400)
    q[B] = m.predict(X[B])
    dec = json.loads((ranker.ROOT / "models" / "decision_na.json").read_text())
    rule = (rt["p1"].to_numpy() >= dec["thr_addr"]) & (rt["m12"].to_numpy() >= dec["margin"])
    print(f"address records: {rt.height:,} held-out ({int(own.sum()):,} owned); acceptance threshold on q chosen on the thr half: {best[0]:.2f}")
    for name, acc in (("current rule (p1 >= %.2f, margin %.2f)" % (dec["thr_addr"], dec["margin"]), rule), ("record-level accept model", q >= best[0])):
        a, b = score(acc, A, n_own_addr[False]), score(acc, B, n_own_addr[True])
        print(f"  {name:40s} thr half {a['official']:.4f}   REPORT half {b['official']:.4f}  (report: TP {b['tp']}, FP wrong {b['fp_wrong']}, FP orphan-w {b['fp_orph_w']:.0f})")
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", ", ".join(k for k, _ in imp[:10]))


if __name__ == "__main__":
    main()
