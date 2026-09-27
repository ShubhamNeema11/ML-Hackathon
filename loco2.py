"""Leave-one-country-out, part 2 (see loco.py): what else generalizes to an unseen country (India, model trained on US only)?
  a) no country codes AND no language / country-dependent features (name lengths, state / city equality flags)
  b) no country codes, decision thresholds tuned on India itself (oracle): how much of the unseen-country gap is threshold calibration"""
import polars as pl
import loco
import ranker
from ranker import FEATURES, NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, record_table, tune_decision

LANG = ["q_name_len", "s_name_len", "state_eq", "state_conflict", "city_eq"]


def main():
    cty = loco.country_of()
    tr = pl.read_parquet(NORM / f"feat_train{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "label"]).join(cty, on="rec")
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "s1", "label"]).join(cty, on="rec")
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"}).join(cty, on="rec")
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi").join(cty, on="rec")
    ow = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    us = tr.filter(pl.col("country") == "US")

    def run(cols, label, tune_on):
        m = loco.fit(us, cols)
        p = m.predict(ev.select(cols).cast(pl.Float32).to_numpy())
        rt = record_table(ev, p).join(ev.select("rec", "country").unique("rec"), on="rec").with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("es"))
        half = lambda c, es: rt.filter((pl.col("country") == c) & (pl.col("es") == es))
        n_ = lambda c, es: owned.filter((pl.col("country") == c) & ((pl.col("rec").hash(seed=3) % 2 == 0) == es)).height
        t = half(tune_on, False)
        dec = tune_decision(t, owned.join(t.select("rec"), on="rec", how="semi"), n_(tune_on, False), ow, "")
        r_ = half("India", True)
        r = apply_decision(r_, owned.join(r_.select("rec"), on="rec", how="semi"), n_("India", True), ow, dec)
        print(f"RESULT {label:62s} India {expected_official(r['tp'], n_('India', True), r['fp_wrong_owner'] + r['fp_orphan_weighted']):.4f} "
              f"(P {r['precision']:.4f} R {r['recall']:.4f})  thresholds {dec['thr_addr']}/{dec['thr_noaddr']}/{dec['margin']}", flush=True)

    nocodes = [c for c in FEATURES if c not in loco.CODES]
    run([c for c in nocodes if c not in LANG], "a) no codes, no name-length / state / city flags (US thresholds)", "US")
    run(nocodes, "b) no codes, thresholds tuned on INDIA (oracle calibration)", "India")


if __name__ == "__main__":
    main()
