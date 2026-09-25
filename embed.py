"""Dense retrieval channel: fine-tuned multilingual-e5-small (MIT, 118M params).

Text = native-script normalized name | normalized address, so the model sees the original script
(Devanagari, Tamil, French accents...) and learns transliteration/paraphrase that sparse keys miss.

Usage:
  python embed.py train                 fine-tune on (S2/S3 record, S1 owner) pairs, held-out S1 excluded
  python embed.py eval                  recall@k on the same held-out queries as `block.py eval`, + union with sparse
  python embed.py search train|test     full run -> normalized/[test_]candidates_dense.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from block import NORM, ROOT, TOP_K, ground_truth, is_val_s1, recall_report

BASE_MODEL = "intfloat/multilingual-e5-small"
MODEL_DIR = ROOT / "models" / "e5_er"
MAX_LEN = 64
N_TRAIN = 600_000


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
    t = pl.concat([texts("train", i) for i in (1, 2, 3)]).select("entity_id", "text")
    df = (pairs.join(t.rename({"entity_id": "rec", "text": "anchor"}), on="rec")
               .join(t.rename({"entity_id": "s1", "text": "positive"}), on="s1").select("anchor", "positive"))
    print(f"training pairs: {df.height:,}", flush=True)
    ds = Dataset.from_polars(df)

    model = SentenceTransformer(BASE_MODEL, device="cuda")
    model.max_seq_length = MAX_LEN
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=128)  # big effective batch on 6 GB
    args = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_DIR / "ckpt"), num_train_epochs=1, per_device_train_batch_size=512,
        learning_rate=5e-5, warmup_ratio=0.05, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100, save_strategy="no", dataloader_num_workers=0, report_to="none")
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    model.save(str(MODEL_DIR))
    print("saved", MODEL_DIR, flush=True)


def encode(model, txt: list[str]) -> torch.Tensor:
    # sort by length so each batch pads little; restore order afterwards
    order = np.argsort([len(s) for s in txt])
    emb = model.encode([txt[i] for i in order], batch_size=1024, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False).half()
    out = torch.empty_like(emb)
    out[torch.as_tensor(order, device=emb.device)] = emb
    return out


def search(s1: pl.DataFrame, q: pl.DataFrame, model, top_k: int = TOP_K) -> pl.DataFrame:
    """Exact top-k cosine per country on GPU. (rec, s1, dense_score, dense_rank)."""
    out = []
    for c in q["country"].unique().to_list():
        s = s1.filter(pl.col("country") == c)
        qq = q.filter(pl.col("country") == c)
        if s.height == 0 or qq.height == 0:
            continue
        t = time.time()
        es = encode(model, s["text"].to_list())
        eq = encode(model, qq["text"].to_list())
        print(f"  {c}: encoded S1={s.height:,} q={qq.height:,} in {time.time() - t:.0f}s", flush=True)
        k = min(top_k, s.height)
        sc, ix = [], []
        for i in range(0, eq.shape[0], 512):
            v, j = torch.topk(eq[i:i + 512] @ es.T, k, dim=1)
            sc.append(v.float().cpu()); ix.append(j.cpu())
        sc, ix = torch.cat(sc).numpy(), torch.cat(ix).numpy()
        sid = s["entity_id"].to_numpy()
        out.append(pl.DataFrame({
            "rec": np.repeat(qq["entity_id"].to_numpy(), k), "s1": sid[ix.ravel()],
            "dense_score": sc.ravel(), "dense_rank": np.tile(np.arange(1, k + 1, dtype=np.uint16), qq.height)}))
        del es, eq
        torch.cuda.empty_cache()
    return pl.concat(out)


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
        q = pl.concat([texts("train", 2), texts("train", 3)]).join(qid, on="entity_id", how="semi")
        truth = ground_truth().filter(is_val_s1()).join(qid.rename({"entity_id": "rec"}), on="rec", how="semi")
        dense = search(s1, q, load_model())
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
        split = sys.argv[2]
        s1 = texts(split, 1)
        q = pl.concat([texts(split, 2), texts(split, 3)])
        dense = search(s1, q, load_model())
        dense.write_parquet(NORM / f"{'test_' if split == 'test' else ''}candidates_dense.parquet")
