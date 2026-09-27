"""Monotonic constraints for robustness to shifted data: more similarity may only RAISE the match probability, more conflict / gap only
LOWER it. Leave-one-country-out as loco.py (train US only, India unseen; codes dropped, the best known generic setting)."""
import polars as pl
import lightgbm as lgb
import numpy as np
import loco
import ranker
from ranker import FEATURES, NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, record_table, tune_decision

UP = ["sparse_score", "dense_score", "both", "dense_margin", "name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw", "name_native_ratio",
      "addr_tset", "addr_partial", "num_common", "num_jaccard", "num_equal", "num_subset", "unit_common", "rare_common", "legal_same", "ce_score"]
DOWN = ["sparse_rank", "dense_rank", "dense_gap_best", "sparse_gap_best", "num_only_q", "num_only_s", "num_conflict", "num_absdiff", "unit_only_q",
        "unit_only_s", "unit_conflict", "rare_only_q", "rare_only_s", "legal_conflict", "state_conflict", "ce_gap_best"]


def fit(tr, cols, mono):
    y = tr["label"].to_numpy()
    rng = np.random.default_rng(0)
    keep = (y == 1) | (rng.random(len(y)) < 0.15)
    w = np.where(y == 1, 1.0, 1 / 0.15)[keep]
    x = tr.select(cols).cast(pl.Float32).to_numpy()[keep]
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1)
    if mono:
        params.update(monotone_constraints=[1 if c in UP else -1 if c in DOWN else 0 for c in cols], monotone_constraints_method="advanced")
    return lgb.train(params, lgb.Dataset(x, y[keep], weight=w, feature_name=cols), 400)


def main():
    cty = loco.country_of()
    tr = pl.read_parquet(NORM / f"feat_train{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "label"]).join(cty, on="rec").filter((pl.col("country") == "US") & (pl.col("rec").hash(seed=41) % 10 < 4))
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "s1", "label"]).join(cty, on="rec")
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"}).join(cty, on="rec")
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi").join(cty, on="rec")
    ow = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    cols = [c for c in FEATURES if c not in loco.CODES]
    for label, mono in (("no codes, 4 conflict flags only (as now)", False), ("no codes, FULL monotonic constraints", True)):
        if not mono:
            print("RESULT reference from loco_ablate: no codes -> India (unseen) 0.9619, US 0.9844 (same 40% sample)", flush=True)
            continue
        m = fit(tr, cols, mono)
        p = m.predict(ev.select(cols).cast(pl.Float32).to_numpy())
        rt = record_table(ev, p).join(ev.select("rec", "country").unique("rec"), on="rec").with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("es"))
        half = lambda c, es: rt.filter((pl.col("country") == c) & (pl.col("es") == es))
        n_ = lambda c, es: owned.filter((pl.col("country") == c) & ((pl.col("rec").hash(seed=3) % 2 == 0) == es)).height
        t = half("US", False)
        dec = tune_decision(t, owned.join(t.select("rec"), on="rec", how="semi"), n_("US", False), ow, "")
        out = {}
        for c in ("US", "India"):
            r_ = half(c, True)
            r = apply_decision(r_, owned.join(r_.select("rec"), on="rec", how="semi"), n_(c, True), ow, dec)
            out[c] = (expected_official(r["tp"], n_(c, True), r["fp_wrong_owner"] + r["fp_orphan_weighted"]), r["precision"], r["recall"])
        print(f"RESULT {label:44s} India (unseen) {out['India'][0]:.4f} (P {out['India'][1]:.4f} R {out['India'][2]:.4f})   US {out['US'][0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
