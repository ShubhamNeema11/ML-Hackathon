"""Leave-one-country-out: how much does an UNSEEN country cost the ranker, and do country-relative (generic) features fix it?

Labelled data only (the ranker-training pairs and the held-out pairs of US / India). France in the test set is an unseen country; here India
plays that role: the ranker and its decision thresholds are learned on US data only and evaluated on India's held-out records.
  raw        B2's features, US-only model                                           (what France gets today)
  relative   every continuous feature replaced by its percentile among the candidate pairs of the SAME country (computed automatically
             for any data, no hand-written country rules), country-specific code features (state / legal codes) dropped
  seen       B2's features, model trained on US + India (India is a seen country: the upper reference)
Report: expected official score on India's held-out report half (thresholds tuned on the US held-out 'thr' half), and on US for reference.

  ER_CE=1 ER_CE_TAG=_v2 ER_FEAT_TAG=_ce2 python loco.py
"""
import numpy as np
import polars as pl
import lightgbm as lgb

import ranker
from ranker import FEATURES, NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, record_table, tune_decision

CODES = ["legal_code_q", "legal_code_s", "state_code_q", "state_code_s"]


def country_of() -> pl.DataFrame:
    return pl.concat([pl.read_parquet(NORM / f"source{i}.parquet", columns=["entity_id", "country"]) for i in (2, 3)]).rename({"entity_id": "rec"})


def continuous(df: pl.DataFrame) -> list:
    return [c for c in FEATURES if c not in CODES and df[c].n_unique() > 20]


def relative(df: pl.DataFrame, cont: list) -> pl.DataFrame:
    """Percentile of each continuous feature inside its own country (ties averaged, nulls stay null)."""
    return df.with_columns([(pl.col(c).rank("average").over("country") / pl.col(c).count().over("country")).cast(pl.Float32).alias(c) for c in cont])


def fit(tr: pl.DataFrame, cols: list):
    y = tr["label"].to_numpy()
    rng = np.random.default_rng(0)
    keep = (y == 1) | (rng.random(len(y)) < 0.15)
    w = np.where(y == 1, 1.0, 1 / 0.15)[keep]
    x = tr.select(cols).cast(pl.Float32).to_numpy()[keep]
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1)
    return lgb.train(params, lgb.Dataset(x, y[keep], weight=w, feature_name=cols), 400)


def main():
    cty = country_of()
    tr = pl.read_parquet(NORM / f"feat_train{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "label"]).join(cty, on="rec")
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "s1", "label"]).join(cty, on="rec")
    print("train pairs by country:", tr.group_by("country").len().to_dicts(), "| eval:", ev.group_by("country").len().to_dicts(), flush=True)
    cont = continuous(tr)
    print(f"{len(cont)} continuous features made country-relative; dropped codes: {CODES}", flush=True)
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"}).join(cty, on="rec")
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi").join(cty, on="rec")
    ow = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)

    def evaluate(model, e, cols, label):
        p = model.predict(e.select(cols).cast(pl.Float32).to_numpy())
        rt = record_table(e, p).join(e.select("rec", "country").unique("rec"), on="rec").with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("es"))
        us_thr = rt.filter((pl.col("country") == "US") & ~pl.col("es"))
        n_us = owned.filter((pl.col("country") == "US") & ((pl.col("rec").hash(seed=3) % 2 == 0) == False)).height
        dec = tune_decision(us_thr, owned.join(us_thr.select("rec"), on="rec", how="semi"), n_us, ow, "")   # thresholds from US only
        out = {}
        for c in ("US", "India"):
            r_ = rt.filter((pl.col("country") == c) & pl.col("es"))
            n_ = owned.filter((pl.col("country") == c) & ((pl.col("rec").hash(seed=3) % 2 == 0) == True)).height
            r = apply_decision(r_, owned.join(r_.select("rec"), on="rec", how="semi"), n_, ow, dec)
            out[c] = (expected_official(r["tp"], n_, r["fp_wrong_owner"] + r["fp_orphan_weighted"]), r["precision"], r["recall"])
        print(f"RESULT {label:44s} India {out['India'][0]:.4f} (P {out['India'][1]:.4f} R {out['India'][2]:.4f})   US {out['US'][0]:.4f}", flush=True)
        return out

    raw_cols = FEATURES
    rel_cols = [c for c in FEATURES if c not in CODES]
    us = tr.filter(pl.col("country") == "US")
    evaluate(fit(us, raw_cols), ev, raw_cols, "raw features, trained on US only (unseen India)")
    tr_rel, ev_rel = relative(tr, cont), relative(ev, cont)
    evaluate(fit(tr_rel.filter(pl.col("country") == "US"), rel_cols), ev_rel, rel_cols, "country-relative features, US only")
    evaluate(fit(tr.filter(pl.col("country") == "US").drop(CODES).with_columns([pl.lit(0).alias(c) for c in CODES]), rel_cols), ev, rel_cols, "raw features without codes, US only")
    evaluate(fit(tr, raw_cols), ev, raw_cols, "raw features, trained on US + India (seen)")


if __name__ == "__main__":
    main()
