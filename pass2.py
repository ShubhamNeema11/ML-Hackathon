"""Pass 2: collective re-ranking with sibling support.

Pass 1 (ranker B2) decides every S2/S3 record alone. Records of one business should agree on one S1 entity, and that is
information pass 1 never sees: how many other records already point to the same S1, how strongly, whether a confident
sibling looks like this record, whether the S1 is already "full". Pass 2 adds exactly those features on top of the
pass-1 probability and re-scores the top candidates of every record.

  python pass2.py fit      ER_ROOT = the mirror folder made by fulltrain_setup.py (training records as a pseudo test set)
  python pass2.py apply    ER_ROOT = the real work folder: reads the pass-1 test scores, writes pass-2 score parts

Why a full-density mirror: sibling features depend on how many records exist for an S1, so training rows must come from a
run over ALL records, exactly like the test run. Rows used for fitting exclude every record any earlier model has seen
(ranker / cross-encoder queries, held-out queries, the embedder's 600k pairs). The evaluation population is the held-out
S1 entities' records plus 10% of the orphans (the real orphan share), split in halves: thresholds are tuned on one half and
the score is reported on the other, next to pass 1 evaluated by the very same procedure.

Environment: ER_PASS1 (dir of pass-1 score parts, default normalized/pred_final), ER_PASS2_OUT (default normalized/pred_pass2),
ER_PASS2_MODEL / ER_PASS2_DECISION (file names under models/), ER_INSAMPLE_DIR (normalized/ of the real training run, holds
train_queries.parquet and eval_queries.parquet), ER_PASS2_FIT_FRAC (share of fit records used, default 0.3).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from block import NORM, ROOT, ground_truth, is_val_s1

MIN_P = 0.005      # pairs below this pass-1 probability are dropped (they never win)
TOPR = 3           # candidates per record that pass 2 re-scores
SIB_P = 0.8        # a record counts as a confident sibling of its top S1 above this probability
N_SIB = 4          # siblings compared per S1
PASS1 = Path(os.environ.get("ER_PASS1", NORM / "pred_final"))
OUTDIR = Path(os.environ.get("ER_PASS2_OUT", NORM / "pred_pass2"))
MODEL_PATH = ROOT / "models" / os.environ.get("ER_PASS2_MODEL", "pass2.txt")
DECISION_PATH = ROOT / "models" / os.environ.get("ER_PASS2_DECISION", "decision_pass2.json")
INSAMPLE = Path(os.environ.get("ER_INSAMPLE_DIR", NORM))
FIT_FRAC = float(os.environ.get("ER_PASS2_FIT_FRAC", 0.3))

FEATS2 = ["p", "logit", "r_rank", "p1", "p2", "gap_best", "margin", "has_addr", "is_s3",
          "n_oth", "n_oth_hi", "n_oth_s2", "n_oth_s3", "sum_p_oth", "pmax_oth", "n_better", "n_cand_e",
          "sib_n", "sib_sim_max", "sib_sim_mean", "sib_addr_n"]


# ------------------------------------------------------------------------------------------------ world
def world() -> dict:
    """Index tables of this work folder: ids (S1 rows first, then S2, S3 - the order predict.py uses) and per-record arrays."""
    from predict import ids_table
    ids = ids_table()
    cols = [pl.read_parquet(NORM / f"test_source{i}.parquet", columns=["name_norm", "has_addr"]) for i in (1, 2, 3)]
    n1, n2 = cols[0].height, cols[1].height
    allc = pl.concat(cols)
    return dict(ids=ids, n1=n1, n2=n2,
                name=np.array(allc["name_norm"].fill_null("").to_list(), dtype=object),
                addr=allc["has_addr"].cast(pl.Int8).to_numpy())


def load_pass1(parts: list[Path] | None = None) -> pl.DataFrame:
    parts = parts or sorted(PASS1.glob("part*.parquet"))
    return pl.concat([pl.read_parquet(p).filter(pl.col("p") >= MIN_P) for p in parts])


# ------------------------------------------------------------------------------------------------ features
def sibling_features(d: pl.DataFrame, w: dict) -> pl.DataFrame:
    """(rec_i, s1_i, sib_n, sib_sim_max, sib_sim_mean, sib_addr_n): how the record's name compares with the confident
    siblings of the S1 it is a candidate for. Only pairs where pass 1 is unsure are compared."""
    claim = (pl.col("r_rank") == 1) & (pl.col("p") >= SIB_P)
    sibs = (d.filter(claim).sort("p", descending=True).group_by("s1_i", maintain_order=True).head(N_SIB)
             .select("s1_i", pl.col("rec_i").alias("sib_i")))
    todo = d.filter((pl.col("r_rank") <= 2) & (pl.col("p") >= 0.02) & (pl.col("p") < 0.995)).select("rec_i", "s1_i")
    j = todo.join(sibs, on="s1_i").filter(pl.col("rec_i") != pl.col("sib_i"))
    schema = {"rec_i": pl.UInt32, "s1_i": pl.UInt32, "sib_n": pl.UInt32, "sib_sim_max": pl.Float32, "sib_sim_mean": pl.Float32, "sib_addr_n": pl.Int64}
    if j.height == 0:
        return pl.DataFrame(schema=schema)
    out = []
    for lo in range(0, j.height, 4_000_000):
        sl = j.slice(lo, 4_000_000)
        a, b = sl["rec_i"].to_numpy(), sl["sib_i"].to_numpy()
        sim = process.cpdist(w["name"][a].tolist(), w["name"][b].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
        out.append(sl.with_columns(pl.Series("sim", sim.astype(np.float32)), pl.Series("sib_addr", w["addr"][b].astype(np.int64))))
    j = pl.concat(out)
    return j.group_by("rec_i", "s1_i").agg(
        pl.len().cast(pl.UInt32).alias("sib_n"), pl.col("sim").max().alias("sib_sim_max"), pl.col("sim").mean().alias("sib_sim_mean"),
        (pl.col("sib_addr") * (pl.col("sim") >= 80)).sum().alias("sib_addr_n"))


def build_features(d: pl.DataFrame, w: dict) -> pl.DataFrame:
    """d = (rec_i, s1_i, p) pass-1 pairs of ALL records. Returns the top-TOPR pairs of every record with pass-2 features."""
    n_s12 = w["n1"] + w["n2"]
    d = d.sort("p", descending=True).with_columns(pl.int_range(1, pl.len() + 1).over("rec_i").cast(pl.UInt8).alias("r_rank"))
    d = d.filter(pl.col("r_rank") <= TOPR)
    d = d.with_columns(pl.col("p").max().over("rec_i").alias("p1"),
                       pl.col("p").filter(pl.col("r_rank") == 2).first().over("rec_i").fill_null(0.0).alias("p2"))
    is_s3 = pl.col("rec_i") >= n_s12
    claim = (pl.col("r_rank") == 1) & (pl.col("p") >= 0.5)
    claim_hi = (pl.col("r_rank") == 1) & (pl.col("p") >= 0.9)
    ent = d.group_by("s1_i").agg(
        claim.sum().alias("n_claim"), claim_hi.sum().alias("n_claim_hi"),
        (claim & ~is_s3).sum().alias("n_claim_s2"), (claim & is_s3).sum().alias("n_claim_s3"),
        pl.col("p").sum().alias("sum_p"), pl.col("p").max().alias("pmax"),
        pl.col("p").sort(descending=True).get(1, null_on_oob=True).fill_null(0.0).alias("pmax2"),
        pl.len().alias("n_cand_e"))
    d = d.join(ent, on="s1_i").with_columns(
        (pl.col("p").rank("min", descending=True).over("s1_i") - 1).alias("n_better"),
        (pl.col("n_claim") - claim.cast(pl.Int64)).alias("n_oth"),
        (pl.col("n_claim_hi") - claim_hi.cast(pl.Int64)).alias("n_oth_hi"),
        (pl.col("n_claim_s2") - (claim & ~is_s3).cast(pl.Int64)).alias("n_oth_s2"),
        (pl.col("n_claim_s3") - (claim & is_s3).cast(pl.Int64)).alias("n_oth_s3"),
        (pl.col("sum_p") - pl.col("p")).alias("sum_p_oth"),
        pl.when(pl.col("p") >= pl.col("pmax")).then(pl.col("pmax2")).otherwise(pl.col("pmax")).alias("pmax_oth"),
        (pl.col("p1") - pl.col("p")).alias("gap_best"), (pl.col("p") - pl.col("p2")).alias("margin"),
        is_s3.cast(pl.Int8).alias("is_s3"),
        pl.Series("has_addr", w["addr"][d["rec_i"].to_numpy()]).cast(pl.Int8),
        (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log().alias("logit"))
    sib = sibling_features(d, w)
    d = d.join(sib, on=["rec_i", "s1_i"], how="left")
    return d.select("rec_i", "s1_i", *FEATS2)


# ------------------------------------------------------------------------------------------------ evaluation helpers
def embed_records() -> pl.DataFrame:
    """The records the embedder was fine-tuned on (same recipe as embed.train)."""
    n_train = 600_000  # = embed.N_TRAIN (embed.py is not imported: it pulls in torch, which must not share a process with polars here)
    return ground_truth().filter(~is_val_s1()).sample(fraction=1.0, shuffle=True, seed=0).unique("s1", keep="first").head(n_train).select("rec")


def populations(feat: pl.DataFrame, w: dict):
    """Attach labels and the fit / held-out flags (needs the ground truth: mirror folder only)."""
    ids = w["ids"]
    gt = ground_truth()
    id_rec = ids.rename({"entity_id": "rec", "idx": "rec_i"})
    id_s1 = ids.rename({"entity_id": "s1", "idx": "s1_i"})
    truth = gt.join(id_rec, on="rec").join(id_s1, on="s1").select("rec_i", "s1_i", "rec", "s1")
    owned_h = gt.filter(is_val_s1()).join(id_rec, on="rec").select("rec_i")
    rec_tab = id_rec.filter(pl.col("rec_i") >= w["n1"]).join(gt.select("rec").with_columns(pl.lit(1).alias("own")), on="rec", how="left")
    orphan_h = rec_tab.filter(pl.col("own").is_null() & (pl.col("rec").hash(seed=7) % 10 == 0)).select("rec_i")
    held = pl.concat([owned_h, orphan_h]).with_columns(pl.lit(True).alias("is_h"))
    extra = [pl.read_parquet(INSAMPLE / "trainall_queries.parquet").rename({"entity_id": "rec"})] if (INSAMPLE / "trainall_queries.parquet").exists() else []  # the no-address specialist's training records
    seen = pl.concat([pl.read_parquet(INSAMPLE / "train_queries.parquet").rename({"entity_id": "rec"}),
                      pl.read_parquet(INSAMPLE / "eval_queries.parquet").rename({"entity_id": "rec"}), embed_records(), *extra]).unique().join(id_rec, on="rec").select("rec_i").with_columns(pl.lit(True).alias("seen"))
    feat = (feat.join(truth.select("rec_i", "s1_i").with_columns(pl.lit(1, pl.Int8).alias("label")), on=["rec_i", "s1_i"], how="left")
                .with_columns(pl.col("label").fill_null(0))
                .join(held, on="rec_i", how="left").join(seen, on="rec_i", how="left")
                .with_columns(pl.col("is_h").fill_null(False), pl.col("seen").fill_null(False)))
    return feat, truth.select("rec_i").unique(), owned_h


def decide_and_report(name: str, rt: pl.DataFrame, owned_h: pl.DataFrame, orphan_w: float = 1.0):
    """Tune on half A of the held-out population, report on half B (the same procedure for any probability column)."""
    import ranker
    halves = {}
    for hn, is_a in (("thr-half", True), ("report-half", False)):
        r = rt.filter((pl.col("rec").hash(seed=3) % 2 == 0) == is_a)
        o = owned_h.rename({"rec_i": "rec"}).filter((pl.col("rec").hash(seed=3) % 2 == 0) == is_a)
        halves[hn] = (r, o.join(r.select("rec"), on="rec", how="semi"), o.height)
    dec = ranker.tune_decision(halves["thr-half"][0], halves["thr-half"][1], halves["thr-half"][2], orphan_w, name)
    held = {}
    for hn, (r, o, n) in halves.items():
        res = ranker.apply_decision(r, o, n, orphan_w, dec)
        held[hn] = dict(res, official=ranker.expected_official(res["tp"], n, res["fp_wrong_owner"] + res["fp_orphan_weighted"]))
    print(f"RESULT {name:10s} thr-half {held['thr-half']['official']:.4f}   report-half {held['report-half']['official']:.4f}", flush=True)
    return dec, held


# ------------------------------------------------------------------------------------------------ fit / apply
def fit():
    import lightgbm as lgb
    t0 = time.time()
    w = world()
    feat = build_features(load_pass1(), w)
    print(f"pass-2 features {feat.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
    feat, _, owned_h = populations(feat, w)
    fit_rows = feat.filter(~pl.col("is_h") & ~pl.col("seen") & (pl.col("rec_i").hash(seed=5) % 1000 < int(FIT_FRAC * 1000)))
    es = (fit_rows["rec_i"].hash(seed=9) % 10 == 0).to_numpy()
    x, y = fit_rows.select(FEATS2).cast(pl.Float32).to_numpy(), fit_rows["label"].to_numpy()
    print(f"fit rows {len(y):,} (positives {int(y.sum()):,}), early-stop rows {int(es.sum()):,}", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=int(os.environ.get("ER_THREADS", os.cpu_count() or 8)))
    m = lgb.train(params, lgb.Dataset(x[~es], y[~es], feature_name=FEATS2), int(os.environ.get("ER_MAX_ROUNDS", 1500)),
                  valid_sets=[lgb.Dataset(x[es], y[es])], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(str(MODEL_PATH))
    h = feat.filter(pl.col("is_h"))
    h = h.with_columns(pl.Series("p2new", m.predict(h.select(FEATS2).cast(pl.Float32).to_numpy())))
    n_owned_pop = owned_h.height

    def table(pcol):  # one row per held-out record: best / second-best probability by `pcol`, label and address flag of the best
        d = h.select(pl.col("rec_i").alias("rec"), "s1_i", "label", pl.col("has_addr").alias("q_has_addr"), pl.col(pcol).alias("pp")).sort("pp", descending=True)
        return d.group_by("rec", maintain_order=True).agg(
            pl.col("pp").first().alias("p1"), pl.col("pp").get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
            pl.col("label").first().alias("label1"), pl.col("q_has_addr").first().alias("has_addr"))
    print(f"held-out population: {n_owned_pop:,} owned records", flush=True)
    dec1, held1 = decide_and_report("pass1", table("p"), owned_h)
    dec2, held2 = decide_and_report("pass2", table("p2new"), owned_h)
    DECISION_PATH.write_text(json.dumps(dict(dec2, held=held2, pass1_held=held1, pass1_decision=dec1)), encoding="utf-8")
    imp = sorted(zip(FEATS2, m.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", ", ".join(k for k, _ in imp[:10]), f"   ({time.time() - t0:.0f}s)", flush=True)
    gain = held2["report-half"]["official"] - held1["report-half"]["official"]
    print(f"pass 2 minus pass 1 on the report half: {gain:+.4f}  ->  {'USE pass 2' if gain > 0.0005 else 'keep pass 1'}", flush=True)


def apply():
    """Pass-1 test scores -> pass-2 score parts (same file names / schema as predict.py score; pairs outside the top candidates get p = 0)."""
    import lightgbm as lgb
    t0 = time.time()
    w = world()
    m = lgb.Booster(model_file=str(MODEL_PATH))
    parts = sorted(PASS1.glob("part*.parquet"))
    feat = build_features(load_pass1(parts), w)
    feat = feat.with_columns(pl.Series("p_new", m.predict(feat.select(m.feature_name()).cast(pl.Float32).to_numpy()), dtype=pl.Float32)).select("rec_i", "s1_i", "p_new")
    print(f"pass-2 scores for {feat.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for i, pth in enumerate(parts):
        out = OUTDIR / pth.name
        if out.exists():
            continue
        d = pl.read_parquet(pth).join(feat, on=["rec_i", "s1_i"], how="left")
        d.select("rec_i", "s1_i", pl.col("p_new").fill_null(0.0).alias("p")).write_parquet(out)
        if i % 20 == 0:
            print(f"  part {i + 1}/{len(parts)} ({time.time() - t0:.0f}s)", flush=True)
    (OUTDIR / "_DONE").touch()
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    {"fit": fit, "apply": apply}[sys.argv[1]]()
