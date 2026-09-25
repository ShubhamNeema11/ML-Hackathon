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
DATASET = Path(os.environ.get("ER_DATASET", r"C:\Users\Lenovo\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"))
TOP_K = 10  # per channel: union of sparse+dense top-10 = 99.2% held-out recall at ~18 cands/record
EVAL_K = 30
NGRAM_SPAN = 28  # 4-grams from the first 31 characters of the compact name
MAX_DF = 400  # S1 keys shared by more entities than this are too generic to join on (they still count in norms)
CHUNK = int(os.environ.get("ER_CHUNK", 50_000))  # queries per search chunk; also the part-file granularity
INDEX_SLICE = 200_000  # S1 rows keyed at a time when building the index (bounds peak memory)
COLS = ["entity_id", "country", "name_core", "addr_latin", "city", "state"]

_LEGAL_RE = r"\b(pvt|ltd|llc|llp|lp|inc|corp|co|plc|pc|opc|sarl|sas|sasu|sa|eurl|sci|snc|ei|and|the|of|et)\b"
_HONORIFIC_RE = r"^(shri|sri|shree|smt|mr|mrs|ms|m s|messrs|dr)\b"


def load(split: str, n: int) -> pl.DataFrame:
    prefix = "" if split == "train" else "test_"
    return pl.read_parquet(NORM / f"{prefix}source{n}.parquet", columns=COLS)


def load_queries(split: str) -> pl.LazyFrame:
    """S2 then S3 records, lazily: chunks are read one at a time instead of holding 10M rows."""
    prefix = "" if split == "train" else "test_"
    return pl.concat([pl.scan_parquet(NORM / f"{prefix}source{i}.parquet").select(COLS) for i in (2, 3)])


def ground_truth() -> pl.DataFrame:
    gt = pl.read_csv(DATASET / "train" / "train_ground_truth.tsv", separator="\t", infer_schema_length=0, quote_char=None)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("rec")))


def is_val_s1(col: str = "s1") -> pl.Expr:
    """Held-out S1 entities (10%): never used to train the embedding model or the ranker."""
    return pl.col(col).hash(seed=7) % 10 == 0


def eval_queries(q: pl.LazyFrame, pairs: pl.DataFrame, frac: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sampled queries owned by held-out S1 entities, plus orphans. Returns (queries, truth)."""
    q = q.join(pairs.lazy(), left_on="entity_id", right_on="rec", how="left")
    samp = pl.col("entity_id").hash(seed=11) % 100_000 < int(frac * 100_000)
    q = q.filter((is_val_s1() | pl.col("s1").is_null()) & samp).drop("s1").collect()
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
    """Explode records into (row, key-hash) rows; row is the position inside df."""
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
    return (df.with_row_index("row").select("row", pl.col("country").alias("c"), allk).explode("key").drop_nulls("key")
              .select("row", (pl.col("c") + "|" + pl.col("key")).hash(seed=0).alias("key")))  # UInt64: ~4x smaller than strings


class SparseIndex:
    """IDF-weighted key index over S1. Keys are 64-bit hashes and S1 records are row numbers, so 2.2M records
    x ~34 keys stay around 1-2 GB (strings would need >10 GB)."""

    def __init__(self, s1: pl.DataFrame):
        self.ids = s1["entity_id"]
        k1 = pl.concat([keys(s1.slice(i, INDEX_SLICE)).with_columns(pl.col("row") + i)
                        for i in range(0, s1.height, INDEX_SLICE)])
        df = k1.group_by("key").len().rename({"len": "df"})
        df = df.with_columns((pl.lit(s1.height).log() - pl.col("df").log()).cast(pl.Float32).alias("idf"))
        self.norm = (k1.join(df, on="key").group_by("row")
                       .agg((pl.col("idf") ** 2).sum().sqrt().alias("nrm")).rename({"row": "s1_row"}))
        self.post = k1.join(df.filter(pl.col("df") <= MAX_DF), on="key").select(pl.col("row").alias("s1_row"), "key", "idf")

    def search_chunk(self, q: pl.DataFrame, top_k: int) -> pl.DataFrame:
        j = (keys(q).join(self.post, on="key")
             .group_by("row", "s1_row").agg((pl.col("idf") ** 2).sum().alias("w"))
             .join(self.norm, on="s1_row").select("row", "s1_row", (pl.col("w") / pl.col("nrm")).alias("sparse_score")))
        j = j.sort("sparse_score", descending=True).group_by("row", maintain_order=True).head(top_k)
        j = j.with_columns(pl.int_range(1, pl.len() + 1).over("row").cast(pl.UInt16).alias("sparse_rank"))
        return j.select(pl.Series("rec", q["entity_id"].gather(j["row"].to_numpy())),
                        pl.Series("s1", self.ids.gather(j["s1_row"].to_numpy())), "sparse_score", "sparse_rank")

    def search(self, q: pl.DataFrame | pl.LazyFrame, top_k: int = TOP_K, out_dir: Path | None = None) -> pl.DataFrame | None:
        """(rec, s1, sparse_score, sparse_rank) of the top-k S1 candidates per query record.
        With out_dir every chunk becomes its own parquet part and nothing is kept in memory."""
        q = q.lazy()
        n_rows = q.select(pl.len()).collect().item()
        out = []
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(range(0, n_rows, CHUNK)):
            j = self.search_chunk(q.slice(i, CHUNK).collect(), top_k)
            if out_dir is None:
                out.append(j)
                continue
            j.write_parquet(out_dir / f"part{n:04d}.parquet")
            if n % 20 == 0:
                print(f"  sparse {min(i + CHUNK, n_rows):,}/{n_rows:,}", flush=True)
        if out_dir is not None:
            (out_dir / "_DONE").touch()  # marks a complete run for main.py
        return pl.concat(out) if out_dir is None else None


def recall_report(cand: pl.DataFrame, truth: pl.DataFrame, rank_col: str, label: str, ks=(1, 5, 10, 20, 30)):
    r = truth.join(cand, on=["rec", "s1"], how="left")[rank_col].fill_null(10_000)
    print(f"{label:24s} " + "  ".join(f"@{k}:{(r <= k).mean():.4f}" for k in ks), flush=True)


def ranker_train_queries(q: pl.LazyFrame, pairs: pl.DataFrame, frac: float) -> pl.DataFrame:
    """Queries for ranker training: owned by non-held-out S1, or orphans; disjoint from the eval queries."""
    q = q.join(pairs.lazy(), left_on="entity_id", right_on="rec", how="left")
    samp = pl.col("entity_id").hash(seed=13) % 100_000 < int(frac * 100_000)
    q = q.filter((~is_val_s1() | pl.col("s1").is_null()) & samp).drop("s1").collect()
    ev = NORM / "eval_queries.parquet"
    return q.join(pl.read_parquet(ev), on="entity_id", how="anti") if ev.exists() else q


if __name__ == "__main__":
    # eval [frac]      -> eval_sparse.parquet (top EVAL_K) + eval_queries.parquet
    # train [frac]     -> cand/train_sparse/  ranker-training queries (default 10%)
    # test             -> cand/test_sparse/   every test query
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    t = time.time()
    split = "test" if mode == "test" else "train"
    s1 = load(split, 1)
    q = load_queries(split)
    if mode == "eval":
        frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
        q, truth = eval_queries(q, ground_truth(), frac)
        print(f"eval: full S1={s1.height:,}  queries={q.height:,}  (owned {truth.height:,})", flush=True)
    elif mode == "train":
        q = ranker_train_queries(q, ground_truth(), float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
        print(f"ranker-train queries: {q.height:,}", flush=True)
        q.select("entity_id").write_parquet(NORM / "train_queries.parquet")
    idx = SparseIndex(s1)
    del s1
    print(f"index built {time.time() - t:.0f}s", flush=True)
    if mode == "eval":
        cand = idx.search(q, top_k=EVAL_K)
        recall_report(cand, truth, "sparse_rank", "sparse")
        cand.write_parquet(NORM / "eval_sparse.parquet")
        q.select("entity_id").write_parquet(NORM / "eval_queries.parquet")
    else:
        idx.search(q, out_dir=NORM / "cand" / f"{mode}_sparse")
    print(f"done {time.time() - t:.0f}s", flush=True)
