"""Quick go/no-go for a cross-encoder pilot, no LightGBM needed: AUC and hard-pair errors of two cross-encoders on the SAME
held-out pairs (normalized/ce_eval<tag>.parquet joined to the labels in feat_eval).

    ER_ROOT=/data/er_work python aws/compare_ce.py _v2 _pilot
"""
import os
import sys
from pathlib import Path

import json
import polars as pl

N = Path(os.environ.get("ER_ROOT", ".")) / "normalized"
ref, new = sys.argv[1], sys.argv[2]
lab = pl.read_parquet(N / f"feat_eval{os.environ.get('ER_FEAT_TAG', '_ce2')}.parquet", columns=["rec", "s1", "label"])
a = pl.read_parquet(N / f"ce_eval{ref}.parquet").rename({"ce_score": "a"})
b = pl.read_parquet(N / f"ce_eval{new}.parquet").rename({"ce_score": "b"})
d = lab.join(a, on=["rec", "s1"]).join(b, on=["rec", "s1"])
y = d["label"].to_numpy()


def auc(x):
    r = pl.Series(x).rank("average").to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def top1(col):
    t = d.sort(col, descending=True).group_by("rec", maintain_order=True).head(1)
    return t["label"].mean()


print(f"{d.height:,} pairs scored by both, positives {int(y.sum()):,}")
for name, col in ((ref, "a"), (new, "b")):
    x = d[col].to_numpy()
    print(f"{name:8s} AUC {auc(x):.5f}   1-AUC {1 - auc(x):.5f}   top-1 correct {top1(col):.4f}   "
          f"wrong-but-confident (label 0, score>0.9): {int(((y == 0) & (x > 0.9)).sum()):,}   missed (label 1, score<0.5): {int(((y == 1) & (x < 0.5)).sum()):,}")
# gate (heuristic): the pilot is undertrained (under one epoch), so "not clearly worse" is enough to justify the full run
res = {c: dict(one_minus_auc=1 - auc(d[c].to_numpy()), top1=top1(c)) for c in ("a", "b")}
go = res["b"]["one_minus_auc"] <= 1.15 * res["a"]["one_minus_auc"] and res["b"]["top1"] >= res["a"]["top1"] - 0.005
(N.parent / "logs").mkdir(exist_ok=True)
(N.parent / "logs" / "pilot_gate.json").write_text(json.dumps(dict(ref=ref, new=new, go=bool(go), pairs=d.height, **{f"{k}_{m}": v for k, r in res.items() for m, v in r.items()})))
print("GATE:", "GO (the pilot is at least on par with the small cross-encoder after under one epoch)" if go else "NO-GO (the pilot is clearly worse than the small cross-encoder; the full run is not started automatically)")
