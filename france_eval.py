"""Offline evaluation of the French fixes on LABELLED data (there are no French labels).

Each French fix corrects a mechanism; the cost of that mechanism is measured on the labelled US / India held-out records by damaging them the
way French records are damaged (proportions measured on 20,000 confident French test matches), rebuilding their features, scoring them with
the unchanged B2 and applying B2's decision rule (same protocol as everywhere: report half = the half no threshold was tuned on).
The score lost is what the fix gives back on French records.

  scenario 'addr'    France2 fix: query addresses get region / departement-like extra tokens (65%), abbreviated street words (25%), No prefixes (14%)
  scenario 'name'    France3 fix: 16% of queries with a legal form get it dotted inside the name and lose legal_form
  scenario 'generic' France3 risk: the generic-word list is widened (US 43% -> ~63% token coverage, like the French list) for ALL rows

  ER_CE=1 ER_CE_TAG=_v2 python france_eval.py addr name generic
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import ranker
import structfeat
from ranker import NORM, REAL_ORPHAN_SHARE, apply_decision, expected_official, ground_truth, record_table

ABBR = {"street": "st", "road": "rd", "avenue": "av", "boulevard": "bd", "lane": "ln", "drive": "dr", "nagar": "ngr", "colony": "col"}


def damage_addr(addr: pl.Expr, key: pl.Expr) -> pl.Expr:
    h = key.hash(seed=101) % 1000
    a = addr.fill_null("")
    for full, short in ABBR.items():
        a = pl.when((h % 4 == 0) & (a != "")).then(a.str.replace_all(rf"\b{full}\b", short)).otherwise(a)   # 25%
    a = pl.when((h % 7 == 0) & (a != "")).then(pl.lit("no ") + a).otherwise(a)                           # ~14%
    return pl.when(((h // 10) % 20 < 13) & (a != "")).then(a + pl.lit(", district")).otherwise(a)       # 65%


def build_eval(texts_fn) -> pl.DataFrame:
    c = ranker.retrieval_features(ranker.candidates("eval"))
    need = pl.concat([c.select(pl.col("rec").alias("entity_id")), c.select(pl.col("s1").alias("entity_id"))]).unique()
    texts = pl.concat([texts_fn(i).join(need, on="entity_id", how="semi") for i in (1, 2, 3)])
    f = pl.concat([ranker.string_features(c.slice(i, 1_000_000), texts) for i in range(0, c.height, 1_000_000)])
    f = ranker.attach_ce(f, "eval")
    return f.join(ground_truth().with_columns(pl.lit(1, pl.Int8).alias("label")), on=["rec", "s1"], how="left").with_columns(pl.col("label").fill_null(0))


def score(f: pl.DataFrame, label: str):
    b2 = lgb.Booster(model_file=str(ranker.ROOT / "models" / "ranker_b2.txt"))
    dec = json.loads((ranker.ROOT / "models" / "decision_b2.json").read_text())
    qid = pl.read_parquet(NORM / "eval_queries.parquet").rename({"entity_id": "rec"})
    owned = ground_truth().filter(pl.col("s1").hash(seed=7) % 10 == 0).join(qid, on="rec", how="semi")
    ow = (REAL_ORPHAN_SHARE / (1 - REAL_ORPHAN_SHARE)) / ((qid.height - owned.height) / owned.height)
    p = b2.predict(f.select(b2.feature_name()).cast(pl.Float32).to_numpy())
    rt = record_table(f, p).with_columns((pl.col("rec").hash(seed=3) % 2 == 0).alias("es"))
    out = []
    for es in (False, True):
        r_ = rt.filter(pl.col("es") == es); n_ = owned.filter((pl.col("rec").hash(seed=3) % 2 == 0) == es).height
        r = apply_decision(r_, owned.join(r_.select("rec"), on="rec", how="semi"), n_, ow, dec)
        out.append((expected_official(r["tp"], n_, r["fp_wrong_owner"] + r["fp_orphan_weighted"]), r["tp"], r["fp_wrong_owner"], r["fp_orphan_weighted"]))
    print(f"RESULT {label:58s} tuned {out[0][0]:.4f}   REPORT {out[1][0]:.4f}  (report: TP {out[1][1]}, FP wrong {out[1][2]}, FP orphan-w {out[1][3]:.0f})", flush=True)
    return out[1][0]


def main():
    os.environ["ER_FR_ADDR"] = os.environ["ER_FR_NAME"] = os.environ["ER_FR_GENERIC"] = "0"   # no France in eval: keep the plain path
    qset = set(pl.read_parquet(NORM / "eval_queries.parquet")["entity_id"].to_list())
    base = lambda i: ranker.read_texts("", i)
    res = {"clean": score(build_eval(base), "clean held-out (US / India, as B2 sees them)")}
    for sc in sys.argv[1:]:
        if sc == "addr":
            def fn(i):
                d = base(i)
                return d if i == 1 else d.with_columns(pl.when(pl.col("entity_id").is_in(list(qset))).then(damage_addr(pl.col("addr_latin"), pl.col("entity_id"))).otherwise(pl.col("addr_latin")).alias("addr_latin"))
            res[sc] = score(build_eval(fn), "French-style ADDRESS damage (what France2 fixes)")
        elif sc == "dense_addr":   # the same address damage, ALSO in the embedder's text (eval dense search re-run by dense_damage.py)
            os.environ["ER_EVAL_DENSE"] = str(NORM / "eval_dense_damaged.parquet")
            def fn(i):
                d = base(i)
                return d if i == 1 else d.with_columns(pl.when(pl.col("entity_id").is_in(list(qset))).then(damage_addr(pl.col("addr_latin"), pl.col("entity_id"))).otherwise(pl.col("addr_latin")).alias("addr_latin"))
            res[sc] = score(build_eval(fn), "French-style address damage in features AND embeddings")
            del os.environ["ER_EVAL_DENSE"]
        elif sc == "name":
            def fn(i):
                d = base(i)
                if i == 1:
                    return d
                hit = pl.col("entity_id").is_in(list(qset)) & (pl.col("legal_form") != "") & (pl.col("entity_id").hash(seed=202) % 100 < 16)
                dotted = pl.col("legal_form").str.replace_all(" ", "").str.split("").list.join(" ")
                return d.with_columns(pl.when(hit).then(pl.col("name_core") + pl.lit(" ") + dotted).otherwise(pl.col("name_core")).alias("name_core"),
                                      pl.when(hit).then(pl.lit("")).otherwise(pl.col("legal_form")).alias("legal_form"))
            res[sc] = score(build_eval(fn), "French-style LEGAL-IN-NAME damage (what France3 adds)")
        elif sc == "generic":
            v = structfeat.load_vocab()
            s1 = pl.read_parquet(NORM / "source1.parquet", columns=["name_core"])
            toks = s1.select(pl.col("name_core").str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= 2)
            vc = toks.group_by("t").len().sort("len", descending=True)
            cov = (vc["len"].cum_sum() / toks.height)
            n = int((cov < 0.63).sum()) + 1
            wide = vc["t"].head(n).to_list()
            print(f"generic list widened: {len(v['generic'])} -> {n} words (S1 token coverage {float(cov[len(v['generic']) - 1]):.2f} -> {float(cov[n - 1]):.2f})", flush=True)
            orig = structfeat.load_vocab
            structfeat.load_vocab = lambda: dict(orig(), generic=wide)
            res[sc] = score(build_eval(base), "generic list WIDENED like the French one (France3 risk)")
            structfeat.load_vocab = orig
    print("\nSUMMARY (report half): " + ", ".join(f"{k} {v:.4f}" for k, v in res.items()))
    fr_share = 0.15
    for k, v in res.items():
        if k != "clean":
            print(f"  {k}: mechanism cost on labelled data {res['clean'] - v:+.4f}  ->  expected effect of its fix on the leaderboard (France ~{fr_share:.0%} of S1) about {(res['clean'] - v) * fr_share:+.4f}"
                  if k != "generic" else f"  generic: widening changes the labelled score by {v - res['clean']:+.4f} (negative = the French generic list is a risk)")


if __name__ == "__main__":
    main()
