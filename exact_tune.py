"""Tune the decision rule on the EXACT competition metric (per-S1 macro F0.5) instead of the simulation, on the training mirror's held-out
S1 entities (training data, never trained on): half A of those S1 tunes, half B reports. Same rule family (threshold per record type +
margin), plus an optional extra: a lower threshold when the record's best S1 already has other accepted records (group support).

  python exact_tune.py [pred_dir]         default fulltrain/normalized/pred_na; writes models/decision_exact.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from block import ground_truth, is_val_s1

MN = Path("fulltrain/normalized")
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else MN / "pred_na"


def load():
    ids = pl.read_parquet(MN / "test_ids.parquet").rename({"entity_id": "e", "idx": "i"})
    n1 = pl.scan_parquet(MN / "test_source1.parquet").select(pl.len()).collect().item()
    addr = pl.concat([pl.read_parquet(MN / f"test_source{i}.parquet", columns=["has_addr"]) for i in (2, 3)])["has_addr"].to_numpy()
    best = []
    for p in sorted(PRED.glob("part*.parquet")):
        d = pl.read_parquet(p).sort("p", descending=True)
        best.append(d.group_by("rec_i", maintain_order=True).agg(pl.col("s1_i").first(), pl.col("p").first().alias("p1"),
                                                                  pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2")))
    b = pl.concat(best)
    gt = ground_truth().join(ids.rename({"e": "rec", "i": "rec_i"}), on="rec").join(ids.rename({"e": "s1", "i": "s1_i"}), on="s1")
    s1 = pl.read_parquet(MN / "test_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"}).join(ids.rename({"e": "s1", "i": "s1_i"}), on="s1")
    held = s1.filter(is_val_s1()).with_columns((pl.col("s1").hash(seed=21) % 2 == 0).alias("half_b"))
    b = b.join(gt.select("rec_i", pl.col("s1_i").alias("true_s1")), on="rec_i", how="left")
    b = b.join(held.select("s1_i", "half_b"), on="s1_i", how="inner")              # records whose best S1 is a held-out S1
    ntrue = gt.join(held.select("s1_i", "half_b"), on="s1_i").group_by("s1_i").len()
    hs = held.join(ntrue, on="s1_i", how="left").with_columns(pl.col("len").fill_null(0))
    pos = {v: k for k, v in enumerate(hs["s1_i"].to_list())}
    return dict(s1pos=np.array([pos[v] for v in b["s1_i"].to_list()]), p1=b["p1"].to_numpy(), m=(b["p1"] - b["p2"]).to_numpy(),
                addr=addr[b["rec_i"].to_numpy() - n1], correct=(b["true_s1"] == b["s1_i"]).fill_null(False).to_numpy(),
                ntrue=hs["len"].to_numpy(), half_b=hs["half_b"].to_numpy())


def score(D, acc, which):
    k = len(D["ntrue"])
    npred = np.bincount(D["s1pos"][acc], minlength=k); tp = np.bincount(D["s1pos"][acc & D["correct"]], minlength=k)
    nt = D["ntrue"]
    prec = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0); rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12), 0.0)
    s = np.where(nt == 0, (npred == 0).astype(float), np.where(npred == 0, 0.0, f))
    sel = D["half_b"] if which == "B" else ~D["half_b"]
    return float(s[sel].mean())


def rule(D, ta, tn, mg, mn):
    return (D["p1"] >= np.where(D["addr"], ta, tn)) & (D["m"] >= np.where(D["addr"], mg, mn))


def main():
    D = load()
    cur = json.loads(Path("models/decision_na.json").read_text())
    c = (cur["thr_addr"], cur["thr_noaddr"], cur["margin"], cur.get("margin_noaddr", cur["margin"]))
    acc = rule(D, *c)
    print(f"current rule {c}: exact metric  half A {score(D, acc, 'A'):.4f}   half B {score(D, acc, 'B'):.4f}", flush=True)
    grid_t = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    grid_m = [0.0, 0.05, 0.1, 0.2, 0.3]
    best = max(((score(D, rule(D, ta, tn, mg, mn), "A"), (ta, tn, mg, mn)) for ta, tn, mg, mn in itertools.product(grid_t, grid_t, grid_m, grid_m)), key=lambda z: z[0])
    acc = rule(D, *best[1])
    print(f"tuned on the EXACT metric (half A) {best[1]}: half A {best[0]:.4f}   half B (not tuned on) {score(D, acc, 'B'):.4f}", flush=True)
    Path("models/decision_exact.json").write_text(json.dumps(dict(zip(("thr_addr", "thr_noaddr", "margin", "margin_noaddr"), best[1]),
                                                                  exact_half_a=best[0], exact_half_b=score(D, acc, "B"))), encoding="utf-8")


if __name__ == "__main__":
    main()
