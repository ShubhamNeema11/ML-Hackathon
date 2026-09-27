"""Judge the fine-tuned reranker (Stage 1, AWS) on held-out data: does its score add information to B2 inside the uncertain band?
Same protocol as the zero-shot test: B2's probability (+ gap) with / without the cross-encoder / reranker scores, a small LightGBM,
5-fold out-of-fold by record, pair AUC / logloss / top-1 accuracy. Uses normalized/ce_eval_rrv.parquet (from stage1_results/).

  ER_CE=1 ER_CE_TAG=_v2 ER_FEAT_TAG=_ce2 python rr_judge.py
"""
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score

ev = pl.read_parquet("normalized/feat_eval_ce2.parquet")
b2 = lgb.Booster(model_file="models/ranker_b2.txt")
ev = ev.with_columns(pl.Series("p", b2.predict(ev.select(b2.feature_name()).cast(pl.Float32).to_numpy())))
rr = pl.read_parquet("normalized/ce_eval_rrv.parquet").rename({"ce_score": "rr"})
d = ev.select("rec", "s1", "label", "p", "ce_score", "q_has_addr").join(rr, on=["rec", "s1"])
extra = {}
for tag, col in (("nmz", "nmz"), ("adz", "adz")):   # the zero-shot name / address scores of this morning, when present
    try:
        extra[col] = pl.read_parquet(f"normalized/ce_eval_{tag}.parquet").rename({"ce_score": col})
        d = d.join(extra[col], on=["rec", "s1"], how="left")
    except Exception:
        pass
d = d.with_columns(*[(pl.col(c).max().over("rec") - pl.col(c)).alias(f"{c}_gap") for c in ["p", "rr", "ce_score"] + list(extra)])
print(f"held-out band: {d.height:,} pairs, {int(d['label'].sum()):,} true, {d['rec'].n_unique():,} records", flush=True)
fold = (d["rec"].hash(seed=1) % 5).to_numpy()
y = d["label"].to_numpy()
sets = {"B2 p only": ["p", "p_gap"], "B2 p + current cross-encoder": ["p", "p_gap", "ce_score", "ce_score_gap"],
        "B2 p + NEW reranker": ["p", "p_gap", "rr", "rr_gap"], "B2 p + current CE + NEW reranker": ["p", "p_gap", "ce_score", "ce_score_gap", "rr", "rr_gap"]}
if extra:
    sets["B2 p + current CE + zero-shot name/addr (this morning)"] = ["p", "p_gap", "ce_score", "ce_score_gap"] + [c for x in extra for c in (x, f"{x}_gap")]
res = {}
for name, cols in sets.items():
    X = d.select(cols).to_numpy().astype(np.float64)
    oof = np.zeros(len(y))
    for k in range(5):
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, verbose=-1), lgb.Dataset(X[fold != k], y[fold != k]), 300)
        oof[fold == k] = m.predict(X[fold == k])
    r = d.select("rec", "label").with_columns(pl.Series("q", oof)).sort("q", descending=True).group_by("rec", maintain_order=True).agg(pl.col("label").first())
    res[name] = log_loss(y, oof)
    print(f"{name:55s} AUC {roc_auc_score(y, oof):.4f}  logloss {res[name]:.4f}  top-1 right {r['label'].mean():.4f}", flush=True)
base = res["B2 p + current cross-encoder"]
new = res["B2 p + current CE + NEW reranker"]
print(f"\nlogloss with the new reranker vs the current cross-encoder alone: {base:.4f} -> {new:.4f} ({(new - base) / base:+.1%})")
print("zero-shot reference this morning: 0.0651 -> 0.0627 (-3.7%), and ranker C built on it gave 0.0000 on the official score")
