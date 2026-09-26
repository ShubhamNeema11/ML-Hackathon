"""Score the test set and write the submission files.

  python predict.py score                 rank every blocked test pair -> normalized/pred/part*.parquet (all pairs + prob)
  python predict.py write <threshold>     -> output/matching_results.tsv and output/candidate_pairs.tsv

`score` is the expensive step and is done once; `write` is cheap so the threshold can be changed freely.
IDs are mapped to UInt32 indices (normalized/test_ids.parquet) so 180M pairs fit comfortably in memory.

Decision rule: every S2/S3 record has at most one owner in S1 (verified on the training ground truth), so each
record keeps only its single best candidate, and only when the ranker's probability reaches the threshold.
"""
import sys
import time
from pathlib import Path

import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from block import CHUNK, NORM, ROOT
from ranker import CE_TAG, DECISION_PATH, MODEL_PATH, add_extras, join_ce, load_extras, merge_channels, read_texts, retrieval_features, string_features

OUT = Path(os.environ.get("ER_OUT", ROOT / "output"))  # ER_OUT: write elsewhere (tests)
PRED = Path(os.environ.get("ER_PRED", NORM / "pred"))


def ids_table() -> pl.DataFrame:
    p = NORM / "test_ids.parquet"
    if p.exists():
        return pl.read_parquet(p)
    ids = pl.concat([pl.read_parquet(NORM / f"test_source{i}.parquet", columns=["entity_id"]) for i in (1, 2, 3)]).with_row_index("idx")
    ids.write_parquet(p)
    return ids


def load_dense(ids: pl.DataFrame) -> pl.DataFrame:
    """All dense candidates as (rec_i, s1_i, score, rank) with UInt32 indices, sorted by rec_i.
    ~100M rows: 1.4 GB as integers, >6 GB as strings. Part files are converted a few at a time."""
    files = sorted((NORM / "cand" / "test_dense").glob("*.parquet"))
    id_rec = ids.rename({"entity_id": "rec", "idx": "rec_i"})
    id_s1 = ids.rename({"entity_id": "s1", "idx": "s1_i"})
    out = []
    for i in range(0, len(files), 12):
        d = pl.concat([pl.read_parquet(f) for f in files[i:i + 12]])
        out.append(d.join(id_rec, on="rec").join(id_s1, on="s1")
                    .select("rec_i", "s1_i", pl.col("dense_score").cast(pl.Float32), "dense_rank"))
    return pl.concat(out).sort("rec_i")


def score():
    t0 = time.time()
    ids = ids_table()
    names = ids["entity_id"]
    n_s1 = pl.scan_parquet(NORM / "test_source1.parquet").select(pl.len()).collect().item()
    q_ids = pl.concat([pl.scan_parquet(NORM / f"test_source{i}.parquet").select("entity_id") for i in (2, 3)]).collect()["entity_id"]
    texts = pl.concat([read_texts("test_", i) for i in (1, 2, 3)])
    dense = load_dense(ids)
    dense_rec = dense["rec_i"]
    extras = load_extras("test")
    print(f"name-only extras: {0 if extras is None else extras.height:,} rows", flush=True)
    model = lgb.Booster(model_file=str(MODEL_PATH))
    cols = model.feature_name()  # works for the old 28/29-feature models and the new ones alike
    ce = pl.read_parquet(NORM / f"ce_test{CE_TAG}.parquet") if os.environ.get("ER_CE", "0") == "1" and (NORM / f"ce_test{CE_TAG}.parquet").exists() else None
    if any(c.startswith("ce_") for c in cols) and ce is None:  # the model was trained with cross-encoder features
        raise SystemExit(f"{MODEL_PATH.name} uses cross-encoder features: set ER_CE=1 and provide normalized/ce_test{CE_TAG}.parquet "
                         "(python crossenc.py score test); scoring without them would silently degrade the predictions")
    if ce is not None and not any(c.startswith("ce_") for c in cols):
        print("note: cross-encoder scores exist but this model does not use them", flush=True)
    print(f"model {MODEL_PATH.name}: {len(cols)} features; cross-encoder scores: {'yes' if ce is not None else 'no'}", flush=True)
    PRED.mkdir(parents=True, exist_ok=True)
    n_parts = -(-len(q_ids) // CHUNK)
    n_sparse = len(list((NORM / "cand" / "test_sparse").glob("part*.parquet")))
    if n_sparse != n_parts:  # part n of the sparse candidates must cover the same records as chunk n here
        raise SystemExit(f"{n_sparse} sparse candidate parts but {n_parts} scoring chunks: ER_CHUNK={CHUNK} differs from the value "
                         "used by `block.py test`; use the same ER_CHUNK for both")
    print(f"{len(q_ids):,} test records in {n_parts} chunks; dense rows {dense.height:,} ({time.time() - t0:.0f}s)", flush=True)
    id_rec = ids.rename({"entity_id": "rec", "idx": "rec_i"})
    id_s1 = ids.rename({"entity_id": "s1", "idx": "s1_i"})
    for n in range(n_parts):
        out = PRED / f"part{n:04d}.parquet"
        if out.exists():  # resumable
            continue
        # chunk n = records [n*CHUNK, (n+1)*CHUNK) of S2||S3, which sit at indices n_s1 + position in `ids`
        lo, hi = n_s1 + n * CHUNK, n_s1 + min((n + 1) * CHUNK, len(q_ids))
        a, b = dense_rec.search_sorted(lo), dense_rec.search_sorted(hi)
        de = dense.slice(a, b - a)
        de = de.select(pl.Series("rec", names.gather(de["rec_i"].to_numpy())), pl.Series("s1", names.gather(de["s1_i"].to_numpy())),
                       "dense_score", "dense_rank")
        sp = pl.read_parquet(NORM / "cand" / "test_sparse" / f"part{n:04d}.parquet")
        c = merge_channels(sp, de)
        ex = None if extras is None else extras.join(c.select("rec").unique(), on="rec", how="semi")
        c = retrieval_features(add_extras(c, ex))
        need = pl.concat([c.select(pl.col("rec").alias("entity_id")), c.select(pl.col("s1").alias("entity_id"))]).unique()
        f = join_ce(string_features(c, texts.join(need, on="entity_id", how="semi")), ce)
        p = model.predict(f.select(cols).cast(pl.Float32).to_numpy())
        f = (f.select("rec", "s1").with_columns(pl.Series("p", p, dtype=pl.Float32))
              .join(id_rec, on="rec").join(id_s1, on="s1").select("rec_i", "s1_i", "p"))
        f.write_parquet(out)
        if n % 10 == 0:
            print(f"  chunk {n + 1}/{n_parts}  ({time.time() - t0:.0f}s)", flush=True)
    (PRED / "_DONE").touch()
    print(f"scored ({time.time() - t0:.0f}s)", flush=True)


def _write_tsv(df: pl.DataFrame, f):
    f.write(df.write_csv(separator="\t", include_header=False, quote_style="never").encode("utf-8"))


def write(threshold: float | None = None):
    """threshold=None -> use the decision rule tuned by ranker.py fit (models/decision.json): a probability threshold
    per record type (with / without address) and a required margin between the best and the second-best candidate.
    A number -> the old rule: best candidate per record if p >= threshold."""
    t0 = time.time()
    dec = None
    if threshold is None:
        dec = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
        print(f"decision rule {dec}", flush=True)
        n_s1_ = pl.scan_parquet(NORM / "test_source1.parquet").select(pl.len()).collect().item()
        has_addr = pl.concat([pl.scan_parquet(NORM / f"test_source{i}.parquet").select("has_addr") for i in (2, 3)]).collect()["has_addr"].to_numpy()
    OUT.mkdir(exist_ok=True)
    ids = ids_table()
    names = ids["entity_id"]
    n_s1 = pl.read_parquet(NORM / "test_source1.parquet", columns=["entity_id"]).height  # S1 rows come first in ids
    parts = sorted(PRED.glob("part*.parquet"))
    # Deadline fallback (ER_FALLBACK_PRED=<dir of an earlier model's score parts>): chunks that were not scored by the current
    # model yet use the earlier model's probabilities with a plain threshold (ER_FALLBACK_THR, default 0.80).
    fb_dir = os.environ.get("ER_FALLBACK_PRED")
    fb_thr = float(os.environ.get("ER_FALLBACK_THR", 0.80))
    fb_parts = []
    if fb_dir:
        have = {q.name for q in parts}
        fb_parts = [q for q in sorted(Path(fb_dir).glob("part*.parquet")) if q.name not in have]
        print(f"fallback: {len(parts)} chunks from the current model, {len(fb_parts)} chunks from {fb_dir} (threshold {fb_thr})", flush=True)
    best, pairs = [], []
    for is_fb_group, group in ((False, parts), (True, fb_parts)):
        for p in group:
            is_fb = is_fb_group
            try:
                d = pl.read_parquet(p)
            except Exception:  # a chunk file cut off mid-write (scorer stopped at the deadline): use the fallback for it
                if not fb_dir:
                    raise
                print(f"  {p.name} unreadable, using the fallback scores for it", flush=True)
                d = pl.read_parquet(Path(fb_dir) / p.name)
                is_fb = True
            pairs.append(d.select("s1_i", "rec_i"))
            d = d.sort("p", descending=True)
            best.append(d.group_by("rec_i", maintain_order=True).agg(
                pl.col("s1_i").first(), pl.col("p").first(), pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2")
            ).with_columns(pl.lit(is_fb).alias("is_fb")))
    best = pl.concat(best)
    if dec is None:
        best = best.filter(pl.col("p") >= threshold)
    else:
        adr = has_addr[best["rec_i"].to_numpy() - n_s1_]  # S2/S3 records follow the S1 rows in the id table
        pv, p2v, fbv = best["p"].to_numpy(), best["p2"].to_numpy(), best["is_fb"].to_numpy()
        thr = np.where(adr, dec["thr_addr"], dec["thr_noaddr"])
        keep = np.where(fbv, pv >= fb_thr, (pv >= thr) & ((pv - p2v) >= dec["margin"]))
        best = best.filter(pl.Series(keep))
    print(f"{'rule ' + str(dec) if dec else 'threshold ' + str(threshold)}: {best.height:,} records assigned an S1 owner ({time.time() - t0:.0f}s)", flush=True)

    def lists(pr: pl.DataFrame) -> pl.DataFrame:
        """S1 index -> comma-separated S2/S3 ids, one row for every S1 entity."""
        g = (pr.with_columns(pl.Series("rec", names.gather(pr["rec_i"].to_numpy())))
               .group_by("s1_i").agg(pl.col("rec").sort().str.join(",").alias("ids")))
        allS1 = pl.DataFrame({"s1_i": np.arange(n_s1, dtype=np.uint32)})
        return allS1.join(g, on="s1_i", how="left").with_columns(pl.col("ids").fill_null("")).sort("s1_i")

    m = lists(best.select("s1_i", "rec_i"))
    m = m.with_columns(pl.Series("source1_entity_id", names.gather(m["s1_i"].to_numpy()))).select("source1_entity_id", "ids")
    with open(OUT / "matching_results.tsv", "wb") as f:
        f.write(b"source1_entity_id\tmatched_entity_ids\n")
        _write_tsv(m, f)
    print(f"matching_results.tsv: {m.height:,} S1 rows, {(m['ids'] == '').sum():,} empty ({time.time() - t0:.0f}s)", flush=True)

    allp = pl.concat(pairs).sort("s1_i")
    del pairs
    with open(OUT / "candidate_pairs.tsv", "wb") as f:
        f.write(b"source1_entity_id\tcandidate_entity_ids\n")
        step = 200_000
        for lo in range(0, n_s1, step):
            hi = min(lo + step, n_s1)
            sl = allp.filter((pl.col("s1_i") >= lo) & (pl.col("s1_i") < hi))
            c = lists(sl).filter((pl.col("s1_i") >= lo) & (pl.col("s1_i") < hi))
            c = c.with_columns(pl.Series("source1_entity_id", names.gather(c["s1_i"].to_numpy()))).select("source1_entity_id", "ids")
            _write_tsv(c, f)
    print(f"candidate_pairs.tsv written ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score()
    elif sys.argv[1] == "write":
        write(float(sys.argv[2]) if len(sys.argv) > 2 else None)
