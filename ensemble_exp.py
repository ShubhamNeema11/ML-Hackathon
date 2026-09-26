"""Held-out experiment: does averaging LightGBM B2 with an XGBoost model (same features, same hard-example weights) or with the
cross-encoder score beat B2 alone?  Same protocol as ranker.fit: thresholds tuned on the "thr" half, reported on the other half.

    ER_CE=1 ER_FEAT_TAG=_ce2 ER_CE_TAG=_v2 ER_MODEL=ranker_b2.txt python ensemble_exp.py

Writes models/xgb_b2.json and models/ensemble_result.json. Never touches the submission or the LightGBM model.
"""
import gc
import json
import os
import sys

import numpy as np
import polars as pl

import ranker
from ranker import FEATURES, NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, mine_hard, record_table, tune_decision


def main():
    import lightgbm as lgb
    import xgboost as xgb
    tr = pl.read_parquet(NORM / f"feat_train{ranker.FEAT_TAG}.parquet")
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet")
    x = tr.select(FEATURES).cast(pl.Float32).to_numpy()
    y = tr["label"].to_numpy()
    del tr
    gc.collect()
    params = dict(objective="binary", verbose=-1)
    w, keep = mine_hard(x, y, params)
    x, y = x[keep], y[keep]
    xe = ev.select(FEATURES).cast(pl.Float32).to_numpy()
    ye = ev["label"].to_numpy()
    es = (ev["rec"].hash(seed=3) % 2 == 0).to_numpy()
    dtr = xgb.DMatrix(x, y, weight=w, feature_names=FEATURES)
    des = xgb.DMatrix(xe[es], ye[es], feature_names=FEATURES)
    del x
    gc.collect()
    mono = tuple(-1 if f in ranker.structfeat.CONFLICT_FLAGS else 0 for f in FEATURES)
    xp = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.05, max_depth=8,
              min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_bin=255, monotone_constraints=mono)
    bst = xgb.train(xp, dtr, 2000, evals=[(des, "es")], early_stopping_rounds=50, verbose_eval=100)
    bst.save_model(str(ranker.ROOT / "models" / "xgb_b2.json"))
    p_x = bst.predict(xgb.DMatrix(xe, feature_names=FEATURES), iteration_range=(0, bst.best_iteration + 1))
    p_l = lgb.Booster(model_file=str(ranker.MODEL_PATH)).predict(xe)
    ce = ev["ce_score"].to_numpy().astype(np.float64)
    p_ce = np.where(np.isnan(ce), p_l, ce)

    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    gt = ground_truth()
    owned = gt.filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi")
    orphan_w = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    variants = {"lgb": p_l, "xgb": p_x, "lgb+xgb": (p_l + p_x) / 2, "lgb+ce": (p_l + p_ce) / 2, "lgb+xgb+ce": (p_l + p_x + p_ce) / 3}
    out = {}
    for name, p in variants.items():
        rt = record_table(ev, p).with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("is_es"))
        halves = {}
        for hn, is_es in (("thr-half", False), ("early-stop half", True)):
            r_ = rt.filter(pl.col("is_es") == is_es)
            n_ = owned.filter((pl.col("rec").hash(seed=3) % 2 == 0) == is_es).height
            halves[hn] = (r_, owned.join(r_.select("rec"), on="rec", how="semi"), n_)
        dec = tune_decision(halves["thr-half"][0], halves["thr-half"][1], halves["thr-half"][2], orphan_w, name)
        out[name] = dict(dec=dec)
        for hn, (r_, o_, n_) in halves.items():
            r = apply_decision(r_, o_, n_, orphan_w, dec)
            out[name][hn] = expected_official(r["tp"], n_, r["fp_wrong_owner"] + r["fp_orphan_weighted"])
        print(f"RESULT {name:12s} thr-half {out[name]['thr-half']:.4f}   early-stop half {out[name]['early-stop half']:.4f}", flush=True)
    (ranker.ROOT / "models" / "ensemble_result.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
