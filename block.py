"""Sparse candidate generation: for every S2/S3 record retrieve its top-K S1 candidates.

Retrieval direction is record -> S1 because every S2/S3 record has at most one owner in S1.

Keys (all prefixed by country, so country stays an open set):
  n:  name tokens (len>=3)              p:/q:  4-char prefix / suffix of long name tokens (typos)
  k:  consonant skeleton of name tokens (transliteration: 'sanraaisa' ~ 'sunrise')
  g:  char 4-grams of the space-free name (spacing, handles, domain-style names)
  i:  initials of names with >=3 tokens   x:  city + first name token
  a:  address words   h: house numbers (leading zeros stripped)
  s:  number|street   y: state|street word   z: first name token|street word
Score = cosine over IDF-weighted keys (S1 norm uses all its keys, so long records get no free boost).

Usage:
  python block.py eval [query_frac]   full S1 index, sampled held-out queries -> recall@k
  python block.py train|test          full run -> normalized/[test_]candidates_sparse.parquet
"""
import os
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).parent))  # where normalized/ and models/ live
NORM = ROOT / "normalized"
DATASET = Path(r"C:\Users\Lenovo\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset")
TOP_K = 30
NGRAM_SPAN = 28  # 4-grams from the first 31 characters of the compact name
MAX_DF = 400  # S1 keys shared by more entities than this are too generic to join on (they still count in norms)
CHUNK = 50_000
COLS = ["entity_id", "country", "name_core", "addr_latin", "city", "state"]

_LEGAL_RE = r"\b(pvt|ltd|llc|llp|lp|inc|corp|co|plc|pc|opc|sarl|sas|sasu|sa|eurl|sci|snc|ei|and|the|of|et)\b"
_HONORIFIC_RE = r"^(shri|sri|shree|smt|mr|mrs|ms|m s|messrs|dr)\b"


def load(split: str, n: int) -> pl.DataFrame:
    prefix = "" if split == "train" else "test_"
    return pl.read_parquet(NORM / f"{prefix}source{n}.parquet", columns=COLS)


def ground_truth() -> pl.DataFrame:
    gt = pl.read_csv(DATASET / "train" / "train_ground_truth.tsv", separator="\t", infer_schema_length=0, quote_char=None)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("rec")))


def is_val_s1(col: str = "s1") -> pl.Expr:
    """Held-out S1 entities (10%): never used to train the embedding model or the ranker."""
    return pl.col(col).hash(seed=7) % 10 == 0


def eval_queries(q: pl.DataFrame, pairs: pl.DataFrame, frac: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sampled queries owned by held-out S1 entities, plus orphans. Returns (queries, truth)."""
    q = q.join(pairs, left_on="entity_id", right_on="rec", how="left")
    samp = pl.col("entity_id").hash(seed=11) % 100_000 < int(frac * 100_000)
    q = q.filter((is_val_s1() | pl.col("s1").is_null()) & samp).drop("s1")
    truth = pairs.filter(is_val_s1()).join(q.select(pl.col("entity_id").alias("rec")), on="rec", how="semi")
    return q, truth


def skeleton(e: pl.Expr) -> pl.Expr:
    """Consonant skeleton: bridges Indic transliteration and English spellings."""
    e = (e.str.replace_all("ph", "f").str.replace_all("[cq]", "k").str.replace_all("z", "s")
          .str.replace_all("[aeiouyhw]", "").str.replace_all("[^a-z0-9 ]", ""))
    for ch in "bdfgjklmnprstvx":  # collapse doubled letters (retroflex tt/dd etc.), twice for triples
        e = e.str.replace_all(ch + ch, ch).str.replace_all(ch + ch, ch)
    return e


def keys(df: pl.DataFrame) -> pl.DataFrame:
    """Explode records into (entity_id, key) rows."""
    name = (pl.col("name_core").str.to_lowercase().str.replace_all(_HONORIFIC_RE, " ")
              .str.replace_all(_LEGAL_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars())
    toks = name.str.split(" ")
    long_ = toks.list.eval(pl.element().filter(pl.element().str.len_chars() >= 5))
    sk = skeleton(name).str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ")
    compact = name.str.replace_all(" ", "")
    ini = pl.concat_str([toks.list.get(i, null_on_oob=True).str.slice(0, 1).fill_null("") for i in range(6)])
    atoks = pl.col("addr_latin").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ").str.split(" ")
    words = atoks.list.eval(pl.element().filter((pl.element().str.len_chars() >= 4) & ~pl.element().str.contains(r"^\d+$")))
    nums = atoks.list.eval(pl.element().filter(pl.element().str.contains(r"\d{2,}"))
                           .str.replace_all(r"\D", "").str.strip_chars_start("0"))
    nums = nums.list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    first_word = words.list.get(0, null_on_oob=True)
    parts = [
        toks.list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.eval("n:" + pl.element()),
        long_.list.eval("p:" + pl.element().str.slice(0, 4)),
        long_.list.eval("q:" + pl.element().str.slice(-4)),
        sk.list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.eval("k:" + pl.element()),
        pl.concat_list([compact.str.slice(i, 4) for i in range(NGRAM_SPAN)])
          .list.eval(pl.element().filter(pl.element().str.len_chars() == 4)).list.eval("g:" + pl.element()),
        pl.concat_list([pl.when(toks.list.len() >= 3).then(ini).otherwise(None)]).list.eval(("i:" + pl.element()).drop_nulls()),
        pl.concat_list([pl.concat_str([pl.col("city"), toks.list.first()], separator="|")]).list.eval("x:" + pl.element()),
        words.list.eval("a:" + pl.element()),
        nums.list.eval("h:" + pl.element()),
        pl.concat_list([nums.list.get(0, null_on_oob=True) + "|" + first_word]).list.eval(("s:" + pl.element()).drop_nulls()),
        pl.concat_list([("y:" + pl.col("state") + "|" + words.list.get(i, null_on_oob=True)) for i in range(3)])
          .list.eval(pl.element().drop_nulls()),
        pl.concat_list([toks.list.first() + "|" + first_word]).list.eval(("z:" + pl.element()).drop_nulls()),
    ]
    allk = pl.concat_list(parts).list.unique().alias("key")
    return (df.select("entity_id", pl.col("country").alias("c"), allk).explode("key").drop_nulls("key")
              .select("entity_id", (pl.col("c") + "|" + pl.col("key")).alias("key")))


class SparseIndex:
    def __init__(self, s1: pl.DataFrame):
        k1 = keys(s1)
        df = k1.group_by("key").len().rename({"len": "df"})
        df = df.with_columns((pl.lit(s1.height).log() - pl.col("df").log()).alias("idf"))
        self.norm = (k1.join(df, on="key").group_by("entity_id")
                       .agg((pl.col("idf") ** 2).sum().sqrt().alias("nrm")).rename({"entity_id": "s1"}))
        self.post = (k1.join(df.filter(pl.col("df") <= MAX_DF), on="key")
                       .select(pl.col("entity_id").alias("s1"), "key", "idf"))

    def search(self, q: pl.DataFrame, top_k: int = TOP_K) -> pl.DataFrame:
        """(rec, s1, sparse_score, sparse_rank) for the top-k S1 candidates of every query."""
        out = []
        for i in range(0, q.height, CHUNK):
            j = (keys(q.slice(i, CHUNK)).rename({"entity_id": "rec"}).join(self.post, on="key")
                 .group_by("rec", "s1").agg((pl.col("idf") ** 2).sum().alias("w"))
                 .join(self.norm, on="s1").select("rec", "s1", (pl.col("w") / pl.col("nrm")).alias("sparse_score")))
            j = j.sort("sparse_score", descending=True).group_by("rec", maintain_order=True).head(top_k)
            out.append(j.with_columns(pl.int_range(1, pl.len() + 1).over("rec").cast(pl.UInt16).alias("sparse_rank")))
        return pl.concat(out)


def recall_report(cand: pl.DataFrame, truth: pl.DataFrame, rank_col: str, label: str, ks=(1, 5, 10, 20, 30)):
    r = truth.join(cand, on=["rec", "s1"], how="left")[rank_col].fill_null(10_000)
    print(f"{label:24s} " + "  ".join(f"@{k}:{(r <= k).mean():.4f}" for k in ks), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    t = time.time()
    split = "test" if mode == "test" else "train"
    s1 = load(split, 1)
    q = pl.concat([load(split, 2), load(split, 3)])
    if mode == "eval":
        frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
        q, truth = eval_queries(q, ground_truth(), frac)
        print(f"eval: full S1={s1.height:,}  queries={q.height:,}  (owned {truth.height:,})", flush=True)
    idx = SparseIndex(s1)
    print(f"index built {time.time() - t:.0f}s", flush=True)
    cand = idx.search(q)
    print(f"searched {time.time() - t:.0f}s -> {cand.height:,} pairs", flush=True)
    if mode == "eval":
        recall_report(cand, truth, "sparse_rank", "sparse")
        cand.write_parquet(NORM / "eval_sparse.parquet")
        q.select("entity_id").write_parquet(NORM / "eval_queries.parquet")
    else:
        cand.write_parquet(NORM / f"{'test_' if split == 'test' else ''}candidates_sparse.parquet")
