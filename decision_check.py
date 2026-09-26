"""Decision-rule check under different orphan shares (held-out, no model is trained).

The decision thresholds were tuned assuming the TRAINING orphan share (26% of S2/S3 records have no S1 owner). The test set has about 40%
(two independent estimates: S1 count x matched share x records per S1 = 0.602 owned; our assignments / held-out recall = 0.605 owned).
This script scores the whole held-out set (B2 for address records + the no-address specialist) under a given orphan share:
  - the current decision rule (models/decision_na.json),
  - a rule re-tuned for that share on the 'thr' half, reported on the other half (the same two-half protocol as everywhere else),
and writes the re-tuned rule for the given share to models/decision_na_o<share>.json.

  ER_FEAT_TAG=_ce2 ER_B2=ranker_b2.txt python decision_check.py 0.26 0.40
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import noaddr
import ranker
from ranker import NORM, ground_truth


def held_table():
    b2 = lgb.Booster(model_file=str(ranker.ROOT / "models" / os.environ.get("ER_B2", "ranker_b2.txt")))
    ev = pl.read_parquet(NORM / f"feat_eval{ranker.FEAT_TAG}.parquet")
    ev = ev.with_columns(pl.Series("p", b2.predict(ev.select(b2.feature_name()).cast(pl.Float32).to_numpy())))
    na = pl.read_parquet(NORM / f"feat_noaddr{noaddr.TAGV}_eval.parquet")
    sp = lgb.Booster(model_file=str(noaddr.MODEL))
    na = na.with_columns(pl.Series("p", sp.predict(na.select(noaddr.COLS).cast(pl.Float32).to_numpy())), pl.lit(0).cast(pl.Int8).alias("q_has_addr"))
    ev_addr = ev.filter(pl.col("q_has_addr") == 1)
    rt = pl.concat([ranker.record_table(ev_addr, ev_addr["p"].to_numpy()), ranker.record_table(na, na["p"].to_numpy())])
    return rt.with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("is_es"))


def main():
    shares = [float(x) for x in sys.argv[1:]] or [0.26, 0.40]
    rt = held_table()
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi")
    own = rt["rec"].is_in(owned["rec"].implode()).to_numpy()
    es = rt["is_es"].to_numpy()
    n_half = {k: owned.filter((pl.col("rec").hash(seed=3) % 2 == 0) == k).height for k in (False, True)}
    current = json.loads((ranker.ROOT / "models" / "decision_na.json").read_text())
    cur = {k: current[k] for k in ("thr_addr", "thr_noaddr", "margin", "margin_noaddr")}
    for share in shares:
        w = (share / (1 - share)) / ((qid.height - owned.height) / owned.height)
        wv = np.full(rt.height, w)
        A, B = rt.filter(~pl.Series(es)), rt.filter(pl.Series(es))
        tuned = noaddr.tune_joint(A, own[~es], wv[~es], n_half[False])
        print(f"\norphan share {share:.2f} (orphan weight {w:.3f})")
        for name, dec in (("current rule", cur), ("re-tuned for this share", tuned)):
            ra = noaddr.apply_joint(A, own[~es], wv[~es], n_half[False], dec)
            rb = noaddr.apply_joint(B, own[es], wv[es], n_half[True], dec)
            print(f"  {name:24s} {dec}   tuned half {ra['official']:.4f}   REPORT half {rb['official']:.4f}   "
                  f"(report: TP {rb['tp']}, FP wrong {rb['fp_wrong_owner']}, FP orphan-w {rb['fp_orphan_weighted']:.0f}, precision {rb['precision']:.4f}, recall {rb['recall']:.4f})", flush=True)
        out = ranker.ROOT / "models" / f"decision_na_o{int(round(share * 100))}.json"
        rb = noaddr.apply_joint(B, own[es], wv[es], n_half[True], tuned)
        out.write_text(json.dumps(dict(tuned, orphan_share=share, held_report_half=rb)), encoding="utf-8")
        print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
