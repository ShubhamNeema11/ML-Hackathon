"""The EXACT competition metric (per-S1 macro F0.5, S1 without matches count: 1 if nothing is assigned, else 0) on the held-out S1 entities
of the training mirror (fulltrain/: every training record scored by the pipeline, so the complete true group of each S1 is known).
Nothing is fitted. Compares with the simulated 'expected official' used for tuning and breaks the loss down by error type and group size.

  python exact_metric.py [pred_dir] [decision.json]        defaults: fulltrain/normalized/pred_na, models/decision_na.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from block import ground_truth, is_val_s1

MN = Path("fulltrain/normalized")
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else MN / "pred_na"
DEC = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "models/decision_na.json").read_text())


def main():
    ids = pl.read_parquet(MN / "test_ids.parquet")
    n1 = pl.scan_parquet(MN / "test_source1.parquet").select(pl.len()).collect().item()
    addr = pl.concat([pl.read_parquet(MN / f"test_source{i}.parquet", columns=["has_addr"]) for i in (2, 3)])["has_addr"].to_numpy()
    best = []
    for p in sorted(PRED.glob("part*.parquet")):
        d = pl.read_parquet(p).sort("p", descending=True)
        best.append(d.group_by("rec_i", maintain_order=True).agg(pl.col("s1_i").first(), pl.col("p").first().alias("p1"),
                                                                  pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2")))
    b = pl.concat(best)
    a = addr[b["rec_i"].to_numpy() - n1]
    thr = np.where(a, DEC["thr_addr"], DEC["thr_noaddr"]); mg = np.where(a, DEC["margin"], DEC.get("margin_noaddr", DEC["margin"]))
    acc = (b["p1"].to_numpy() >= thr) & ((b["p1"] - b["p2"]).to_numpy() >= mg)
    pred = b.filter(pl.Series(acc)).select("rec_i", "s1_i").with_columns(pl.Series("has_addr", a[acc]))
    idmap = ids.rename({"entity_id": "e", "idx": "i"})
    gt = ground_truth().join(idmap.rename({"e": "rec", "i": "rec_i"}), on="rec").join(idmap.rename({"e": "s1", "i": "s1_i"}), on="s1")
    s1 = pl.read_parquet(MN / "test_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"}).join(idmap.rename({"e": "s1", "i": "s1_i"}), on="s1")
    held = s1.filter(is_val_s1())                                                    # held-out S1 entities (never trained on)
    T = gt.join(held.select("s1_i"), on="s1_i").select("s1_i", "rec_i").with_columns(pl.lit(1).alias("t"))
    P = pred.join(held.select("s1_i"), on="s1_i").select("s1_i", "rec_i", "has_addr").with_columns(pl.lit(1).alias("pr"))
    both = T.join(P, on=["s1_i", "rec_i"], how="full", coalesce=True).with_columns(pl.col("t").fill_null(0), pl.col("pr").fill_null(0))
    per = both.group_by("s1_i").agg(pl.col("t").sum().alias("n_true"), pl.col("pr").sum().alias("n_pred"), (pl.col("t") * pl.col("pr")).sum().alias("tp"))
    per = held.select("s1_i").join(per, on="s1_i", how="left").fill_null(0)
    tp, nt, npr = per["tp"].to_numpy(), per["n_true"].to_numpy(), per["n_pred"].to_numpy()
    prec = np.where(npr > 0, tp / np.maximum(npr, 1), 0.0); rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12), 0.0)
    score = np.where(nt == 0, (npr == 0).astype(float), np.where(npr == 0, 0.0, f))
    per = per.with_columns(pl.Series("score", score), pl.Series("fp", npr - tp), pl.Series("fn", nt - tp))
    n = per.height
    print(f"held-out S1 entities: {n:,}; EXACT per-S1 macro F0.5 = {score.mean():.4f}   (simulation used for tuning: ~0.9887 on this population)")
    print(f"points lost in total: {(1 - score).sum():,.0f} of {n:,} S1  ({1 - score.mean():.4f})\n")
    cat = (pl.when(pl.col("n_true") == 0).then(pl.lit("1 S1 with NO true match"))
             .when((pl.col("fn") > 0) & (pl.col("fp") > 0)).then(pl.lit("4 missing AND wrong members"))
             .when(pl.col("fn") > 0).then(pl.lit("2 missing members only (FN)"))
             .when(pl.col("fp") > 0).then(pl.lit("3 extra wrong members only (FP)")).otherwise(pl.lit("0 perfect")))
    s = per.with_columns(cat.alias("type")).group_by("type").agg(pl.len().alias("S1"), (1 - pl.col("score")).sum().round(0).alias("points_lost"),
                                                                  pl.col("score").mean().round(3).alias("mean_score")).sort("type")
    print("where the points are lost (by S1 type):"); print(s.with_columns((pl.col("points_lost") / (1 - score).sum()).round(3).alias("share_of_loss")))
    g = per.filter(pl.col("n_true") > 0).with_columns(pl.col("n_true").clip(1, 6).alias("group_size")).group_by("group_size").agg(
        pl.len().alias("S1"), pl.col("score").mean().round(4).alias("mean_score"), (1 - pl.col("score")).sum().round(0).alias("points_lost"),
        (pl.col("fn") > 0).mean().round(3).alias("share_missing_some"), (pl.col("fp") > 0).mean().round(3).alias("share_with_extra")).sort("group_size")
    print("\nS1 with matches, by true group size (6 = 6+):"); print(g)
    z = per.filter(pl.col("n_true") == 0)
    print(f"\nS1 with no true match: {z.height:,} ({z.height / n:.1%}); wrongly given >= 1 record: {(z['n_pred'] > 0).sum():,} -> each scores 0")
    fn_recs = both.filter((pl.col("t") == 1) & (pl.col("pr") == 0)).height
    print(f"true (S1, record) links missed: {fn_recs:,} of {int(nt.sum()):,} ({fn_recs / max(int(nt.sum()), 1):.2%})")


if __name__ == "__main__":
    main()
