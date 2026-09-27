"""Code dropout: keep the country-specific code features (state / legal codes) but set them to 'unknown' (0) on a random share of the
training rows, so the model uses them when known and does not break on unknown ones (an unseen country). Generic, no country rules.
  unseen: model trained on US only, evaluated on India (leave-one-country-out, as loco.py)"""
import numpy as np
import polars as pl
import loco
import ranker
from ranker import FEATURES, NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, record_table, tune_decision


def dropout(df: pl.DataFrame, share: float, seed: int = 20260927) -> pl.DataFrame:   # NOT the negative-sampling seed of loco.fit (0): the same seed drew the same numbers and blanked exactly the kept negatives
    rng = np.random.default_rng(seed)
    m = pl.Series(rng.random(df.height) < share)
    return df.with_columns([pl.when(m).then(0).otherwise(pl.col(c)).cast(df.schema[c]).alias(c) for c in loco.CODES])


def main():
    cty = loco.country_of()
    tr = pl.read_parquet(NORM / f"feat_train{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "label"]).join(cty, on="rec")
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "s1", "label"]).join(cty, on="rec")
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"}).join(cty, on="rec")
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi").join(cty, on="rec")
    ow = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    us = tr.filter(pl.col("country") == "US")
    for share in (0.3, 0.5):
        m = loco.fit(dropout(us, share), FEATURES)
        p = m.predict(ev.select(FEATURES).cast(pl.Float32).to_numpy())
        rt = record_table(ev, p).join(ev.select("rec", "country").unique("rec"), on="rec").with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("es"))
        half = lambda c, es: rt.filter((pl.col("country") == c) & (pl.col("es") == es))
        n_ = lambda c, es: owned.filter((pl.col("country") == c) & ((pl.col("rec").hash(seed=3) % 2 == 0) == es)).height
        t = half("US", False)
        dec = tune_decision(t, owned.join(t.select("rec"), on="rec", how="semi"), n_("US", False), ow, "")
        out = {}
        for c in ("US", "India"):
            r_ = half(c, True)
            r = apply_decision(r_, owned.join(r_.select("rec"), on="rec", how="semi"), n_(c, True), ow, dec)
            out[c] = expected_official(r["tp"], n_(c, True), r["fp_wrong_owner"] + r["fp_orphan_weighted"])
        print(f"RESULT code dropout {share:.0%}, trained on US only: India (unseen) {out['India']:.4f}   US {out['US']:.4f}", flush=True)
    print("reference (loco.py): codes 0.9586 / 0.9850, no codes 0.9620 / 0.9851, India seen 0.9874")


if __name__ == "__main__":
    main()
