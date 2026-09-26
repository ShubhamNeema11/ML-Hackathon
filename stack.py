"""Stacked replacement for the final LightGBM ranker. Nothing upstream is retrained: the base models are the finished LightGBM
(models/ranker_b2.txt) and XGBoost (models/xgb_b2.json, from ensemble_exp.py), the cross-encoder scores are the ones already in the
feature files. Only a small meta-model is fitted, on held-out entities, from the three probabilities.

    ER_CE=1 ER_FEAT_TAG=_ce2 ER_CE_TAG=_v2 python stack.py fit      -> models/stack_b2.pkl + models/decision_stack.json

Held-out entities are split three ways by record: the meta-model is fitted on one third, the decision thresholds are tuned on the
second, and the score is reported on the third that neither saw (stored as "early-stop half" so aws/gates.py and main.py read it as before).
The same split scores the plain LightGBM, so the printed comparison is like for like.

Use it in the pipeline by replacing the model:  ER_MODEL=stack_b2.pkl ER_DECISION=decision_stack.json  (predict.py scores it in place of
the LightGBM booster; the output files and everything downstream, pass 2 included, are unchanged).
"""
import json
import os
import pickle
import sys

import numpy as np
import polars as pl

import ranker
from ranker import FEATURES, NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, record_table, tune_decision

LGB_BASE = os.environ.get("ER_STACK_LGB", "ranker_b2.txt")
XGB_BASE = os.environ.get("ER_STACK_XGB", "xgb_b2.json")
STACK_PATH = ranker.ROOT / "models" / os.environ.get("ER_STACK", "stack_b2.pkl")
META_NAMES = ["l_lgb", "l_xgb", "l_ce", "ce_missing", "l_ce_gap", "q_has_addr"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def meta_features(p_l, p_x, ce, ce_gap, has_addr) -> np.ndarray:
    """The meta-model's inputs. Pairs the cross-encoder did not score (NaN) take the mean of the two base logits and set ce_missing."""
    ce = np.asarray(ce, dtype=np.float64)
    miss = np.isnan(ce)
    l_l, l_x = _logit(p_l), _logit(p_x)
    l_ce = np.where(miss, (l_l + l_x) / 2, _logit(np.where(miss, 0.5, ce)))
    gap = np.where(np.isnan(np.asarray(ce_gap, dtype=np.float64)), 0.0, ce_gap)
    return np.column_stack([l_l, l_x, l_ce, miss.astype(np.float64), gap, np.asarray(has_addr, dtype=np.float64)])


class Stack:
    """Loads the base models and the meta-model; `predict(df)` takes a feature frame carrying the base model's columns."""

    def __init__(self, path):
        import lightgbm as lgb
        import xgboost as xgb
        with open(path, "rb") as f:
            s = pickle.load(f)
        root = ranker.ROOT / "models"
        self.lgb = lgb.Booster(model_file=str(root / s["lgb"]))
        self.xgb = xgb.Booster(model_file=str(root / s["xgb"]))
        self.xgb.set_param({"device": "cpu"})
        self.xgb_rounds = int(s["xgb_rounds"])
        self.meta, self.kind = s["meta"], s["kind"]
        self.cols = self.lgb.feature_name()

    def predict(self, f: pl.DataFrame) -> np.ndarray:
        import xgboost as xgb
        x = f.select(self.cols).cast(pl.Float32).to_numpy()
        p_l = self.lgb.predict(x)
        p_x = self.xgb.predict(xgb.DMatrix(x, feature_names=self.cols), iteration_range=(0, self.xgb_rounds))
        m = meta_features(p_l, p_x, f["ce_score"].to_numpy(), f["ce_gap_best"].to_numpy(), f["q_has_addr"].to_numpy())
        return self.meta.predict_proba(m)[:, 1] if self.kind == "lr" else self.meta.predict(m)


def fit():
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet")
    root = ranker.ROOT / "models"
    bl = lgb.Booster(model_file=str(root / LGB_BASE))
    bx = xgb.Booster(model_file=str(root / XGB_BASE))
    bx.set_param({"device": "cpu"})
    rounds = bx.best_iteration + 1
    cols = bl.feature_name()
    if cols != FEATURES:
        raise SystemExit("the LightGBM base was trained on different features than the current env gives: set ER_CE=1 ER_FEAT_TAG=_ce2 ER_CE_TAG=_v2 (B2)")
    xe = ev.select(cols).cast(pl.Float32).to_numpy()
    p_l = bl.predict(xe)
    p_x = bx.predict(xgb.DMatrix(xe, feature_names=cols), iteration_range=(0, rounds))
    m = meta_features(p_l, p_x, ev["ce_score"].to_numpy(), ev["ce_gap_best"].to_numpy(), ev["q_has_addr"].to_numpy())
    y = ev["label"].to_numpy()
    third = (ev["rec"].hash(seed=5) % 3).to_numpy()
    fit_m, thr_m = third == 0, third == 1

    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi")
    orphan_w = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    in3 = lambda d, k: (d["rec"].hash(seed=5) % 3) == k

    lr = LogisticRegression(C=1.0, max_iter=1000).fit(m[fit_m], y[fit_m])
    gb_params = dict(objective="binary", learning_rate=0.05, num_leaves=8, min_data_in_leaf=100, feature_fraction=1.0, lambda_l2=5.0, verbose=-1,
                     monotone_constraints=[1, 1, 1, 0, 0, 0])
    gb = lgb.train(gb_params, lgb.Dataset(m[fit_m], y[fit_m]), 300)
    candidates = {"lgb (baseline)": (None, p_l), "lgb+xgb+ce mean": (None, (p_l + p_x + np.where(np.isnan(ev["ce_score"].to_numpy()), p_l, ev["ce_score"].to_numpy())) / 3),
                  "stack LR": ("lr", lr.predict_proba(m)[:, 1]), "stack GBM": ("gbm", gb.predict(m))}
    results = {}
    for name, (kind, p) in candidates.items():
        rt = record_table(ev, p)
        parts = {k: (rt.filter(in3(rt, k)), owned.filter(in3(owned, k)).join(rt.filter(in3(rt, k)).select("rec"), on="rec", how="semi"), owned.filter(in3(owned, k)).height)
                 for k in (1, 2)}
        dec = tune_decision(*parts[1], orphan_w, name)
        held = {}
        for hn, k in (("thr-half", 1), ("early-stop half", 2)):
            r = apply_decision(*parts[k], orphan_w, dec)
            held[hn] = dict(r, official=expected_official(r["tp"], parts[k][2], r["fp_wrong_owner"] + r["fp_orphan_weighted"]))
        results[name] = (kind, dec, held)
        print(f"RESULT {name:16s} tuned-third {held['thr-half']['official']:.4f}   report-third {held['early-stop half']['official']:.4f}", flush=True)

    best = max((n for n, v in results.items() if v[0]), key=lambda n: results[n][2]["early-stop half"]["official"])
    kind, dec, held = results[best]
    base = results["lgb (baseline)"][2]["early-stop half"]["official"]
    print(f"chosen meta-model: {best}   report-third {held['early-stop half']['official']:.4f}  vs LightGBM alone {base:.4f}  (gain {held['early-stop half']['official'] - base:+.4f})", flush=True)
    with open(STACK_PATH, "wb") as f:
        pickle.dump(dict(kind=kind, meta=lr if kind == "lr" else gb, lgb=LGB_BASE, xgb=XGB_BASE, xgb_rounds=rounds, names=META_NAMES), f)
    dpath = ranker.ROOT / "models" / os.environ.get("ER_DECISION", "decision_stack.json")  # never the default decision.json
    dpath.write_text(json.dumps(dict(dec, held=held, baseline_report=base, model=best)), encoding="utf-8")
    print(f"wrote {STACK_PATH.name} and {dpath.name}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit()
