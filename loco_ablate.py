"""Which feature GROUP makes the ranker fail on an unseen country? Leave-one-country-out (train US only, test India held-out) with one
group removed at a time (country codes removed in all runs: they are known to hurt). The group whose removal lifts India the most is the
country-bound one. Same fit / decision protocol as loco.py (thresholds from US)."""
import sys
import polars as pl
import loco
import ranker
from ranker import FEATURES, NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, record_table, tune_decision

GROUPS = {
    "dense (embedding)": ["dense_score", "dense_rank", "dense_gap_best", "dense_margin", "both"],
    "sparse (key blocking)": ["sparse_score", "sparse_rank", "sparse_gap_best", "both"],
    "cross-encoder": ["ce_score", "ce_gap_best"],
    "address structure (numbers / units / city / state / postal)": ["num_eq", "num_q_n", "num_s_n", "num_common", "num_only_q", "num_only_s", "num_jaccard",
        "num_equal", "num_subset", "num_conflict", "num_absdiff", "unit_common", "unit_only_q", "unit_only_s", "unit_conflict", "city_eq", "state_eq",
        "postal_eq", "state_conflict"],
    "name words / legal (rare / generic / legal forms)": ["rare_common", "rare_only_q", "rare_only_s", "gen_only_q", "gen_only_s", "legal_same",
        "legal_conflict", "legal_missing_one", "legal_form_eq"],
    "lengths / flags": ["q_name_len", "s_name_len", "any_domain", "from_s3", "n_cands", "extra_rank"],
}


def main():
    only = sys.argv[1:]
    cty = loco.country_of()
    tr = pl.read_parquet(NORM / f"feat_train{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "label"]).join(cty, on="rec").filter((pl.col("country") == "US") & (pl.col("rec").hash(seed=41) % 10 < 4))   # 40% of the US training records (relative comparison)
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet", columns=FEATURES + ["rec", "s1", "label"]).join(cty, on="rec")
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"}).join(cty, on="rec")
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi").join(cty, on="rec")
    ow = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    base = [c for c in FEATURES if c not in loco.CODES]
    for name, drop in [("none (no codes only)", [])] + [(k, v) for k, v in GROUPS.items() if not only or k.split()[0] in only]:
        cols = [c for c in base if c not in drop]
        m = loco.fit(tr, cols)
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
        print(f"RESULT drop {name:58s} India (unseen) {out['India'][0]:.4f} (P {out['India'][1]:.4f} R {out['India'][2]:.4f})   US {out['US'][0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
