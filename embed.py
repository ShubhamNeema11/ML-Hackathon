"""Dense retrieval channel: fine-tuned multilingual-e5-small (MIT, 118M params).

Text = native-script normalized name | normalized address, so the model sees the original script
(Devanagari, Tamil, French accents...) and learns transliteration/paraphrase that sparse keys miss.

Usage:
  python embed.py train                 fine-tune on (S2/S3 record, S1 owner) pairs, held-out S1 excluded
  python embed.py eval                  recall@k on the same held-out queries as `block.py eval`, + union with sparse
  python embed.py search train|test     -> normalized/cand/{train,test}_dense/ (top-10 per query)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from block import EVAL_K, NORM, ROOT, TOP_K, ground_truth, is_val_s1, recall_report

BASE_MODEL = "intfloat/multilingual-e5-small"
MODEL_DIR = ROOT / "models" / "e5_er"
MAX_LEN = 64
N_TRAIN = int(os.environ.get("ER_EMBED_PAIRS", 600_000))   # one (record, owner) pair per S1 entity; a value above the number of S1 entities = all of them


def texts(split: str, n: int) -> pl.DataFrame:
    prefix = "" if split == "train" else "test_"
    d = pl.read_parquet(NORM / f"{prefix}source{n}.parquet", columns=["entity_id", "country", "name_norm", "addr_norm"])
    return d.select("entity_id", "country",
                    pl.concat_str([pl.lit("query: "), pl.col("name_norm"), pl.lit(" | "), pl.col("addr_norm")]).alias("text"))


def train():
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    pairs = ground_truth().filter(~is_val_s1())
    # one record per S1 entity per sample -> no duplicate positives inside a batch
    pairs = pairs.sample(fraction=1.0, shuffle=True, seed=0).unique("s1", keep="first").head(N_TRAIN)
    need = pl.concat([pairs.select(pl.col("rec").alias("entity_id")), pairs.select(pl.col("s1").alias("entity_id"))]).unique()
    t = pl.concat([texts("train", i).join(need, on="entity_id", how="semi") for i in (1, 2, 3)]).select("entity_id", "text")
    df = (pairs.join(t.rename({"entity_id": "rec", "text": "anchor"}), on="rec")
               .join(t.rename({"entity_id": "s1", "text": "positive"}), on="s1").select("anchor", "positive"))
    print(f"training pairs: {df.height:,}", flush=True)
    ds = Dataset.from_polars(df)

    model = SentenceTransformer(BASE_MODEL, device="cuda")
    model.max_seq_length = MAX_LEN
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=int(os.environ.get("ER_MINI_BATCH", 128)))  # 128 on a 6 GB card, 512 on 24 GB
    args = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_DIR / "ckpt"), num_train_epochs=1, per_device_train_batch_size=int(os.environ.get("ER_TRAIN_BATCH", 512)),
        learning_rate=5e-5, warmup_ratio=0.05, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100, save_strategy="no", dataloader_num_workers=0, report_to="none")
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    model.save(str(MODEL_DIR))
    print("saved", MODEL_DIR, flush=True)


def encode(model, txt: list[str]) -> torch.Tensor:
    # sort by length so each batch pads little; restore order afterwards
    order = np.argsort([len(s) for s in txt])
    emb = model.encode([txt[i] for i in order], batch_size=int(os.environ.get("ER_ENCODE_BATCH", 512)), convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False).half()
    out = torch.empty_like(emb)
    out[torch.as_tensor(order, device=emb.device)] = emb
    return out


Q_CHUNK = 200_000
SEARCH_BATCH = int(os.environ.get("ER_SEARCH_BATCH", 64))   # queries per GPU matmul: 128 fits a 6 GB card, 1024 a 24 GB card


def search(s1: pl.DataFrame, queries, model, top_k: int = TOP_K, out_dir: Path | None = None):
    """Exact top-k cosine per country on GPU -> (rec, s1, dense_score, dense_rank).
    `queries(country)` yields query chunks (DataFrames with entity_id, text) so the 10M-record query table is
    never held in memory. S1 of a country is encoded once. With out_dir each chunk is written as a parquet part."""
    out = []
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    part = 0
    for c in sorted(s1["country"].unique().to_list()):
        s = s1.filter(pl.col("country") == c)
        t = time.time()
        es = encode(model, s["text"].to_list())
        sid = s["entity_id"].to_numpy()
        k = min(top_k, s.height)
        print(f"  {c}: S1={s.height:,} encoded in {time.time() - t:.0f}s", flush=True)
        done = 0
        for qq in queries(c):
            eq = encode(model, qq["text"].to_list())
            sc, ix = [], []
            for j in range(0, eq.shape[0], SEARCH_BATCH):
                v, jx = torch.topk(eq[j:j + SEARCH_BATCH] @ es.T, k, dim=1)
                sc.append(v.float().cpu()); ix.append(jx.cpu())
            sc, ix = torch.cat(sc).numpy(), torch.cat(ix).numpy()
            df = pl.DataFrame({
                "rec": np.repeat(qq["entity_id"].to_numpy(), k), "s1": sid[ix.ravel()],
                "dense_score": sc.ravel(), "dense_rank": np.tile(np.arange(1, k + 1, dtype=np.uint16), qq.height)})
            if out_dir is None:
                out.append(df)
            else:
                df.write_parquet(out_dir / f"part{part:04d}.parquet")
                part += 1
                done += qq.height
                print(f"    {c} {done:,} queries  {time.time() - t:.0f}s", flush=True)
            del eq
        del es
        torch.cuda.empty_cache()
    if out_dir is not None:
        (out_dir / "_DONE").touch()  # marks a complete run for main.py
    return pl.concat(out) if out_dir is None else None


def load_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(MODEL_DIR) if MODEL_DIR.exists() else BASE_MODEL, device="cuda")
    m.max_seq_length = MAX_LEN
    m.half()
    return m


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "eval":
        t0 = time.time()
        s1 = texts("train", 1)
        qid = pl.read_parquet(NORM / "eval_queries.parquet")
        q = pl.concat([texts("train", i).join(qid, on="entity_id", how="semi") for i in (2, 3)])
        truth = ground_truth().filter(is_val_s1()).join(qid.rename({"entity_id": "rec"}), on="rec", how="semi")
        dense = search(s1, lambda c: [q.filter(pl.col("country") == c)], load_model(), top_k=EVAL_K)
        print(f"dense search {time.time() - t0:.0f}s", flush=True)
        dense.write_parquet(NORM / "eval_dense.parquet")
        sparse = pl.read_parquet(NORM / "eval_sparse.parquet")
        recall_report(sparse, truth, "sparse_rank", "sparse")
        recall_report(dense, truth, "dense_rank", "dense")
        for k in (10, 15, 20, 30):
            u = pl.concat([sparse.filter(pl.col("sparse_rank") <= k).select("rec", "s1"),
                           dense.filter(pl.col("dense_rank") <= k).select("rec", "s1")]).unique()
            hit = truth.join(u, on=["rec", "s1"], how="semi").height / truth.height
            print(f"union top{k:>2} each: recall {hit:.4f}   cands/query {u.height / q.height:.1f}", flush=True)
    elif cmd == "search":
        # train: the ranker-training queries chosen by `block.py train`; test: every test query
        split = sys.argv[2]
        s1 = texts(split, 1)
        keep = pl.read_parquet(NORM / "train_queries.parquet") if split == "train" else None

        def queries(country):
            for n in (2, 3):  # one source at a time
                d = texts(split, n).filter(pl.col("country") == country)
                if keep is not None:
                    d = d.join(keep, on="entity_id", how="semi")
                for i in range(0, d.height, Q_CHUNK):
                    yield d.slice(i, Q_CHUNK)

        search(s1, queries, load_model(), out_dir=NORM / "cand" / f"{split}_dense")
