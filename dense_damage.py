"""Eval dense search with the French-style address damage of france_eval.damage_addr applied to the QUERY texts the embedder reads
(S1 texts unchanged) -> normalized/eval_dense_damaged.parquet. Torch only (no polars-heavy work in this process)."""
import polars as pl
import embed
from block import EVAL_K, NORM
import france_eval

s1 = embed.texts("train", 1)
qid = pl.read_parquet(NORM / "eval_queries.parquet")
q = pl.concat([pl.read_parquet(NORM / f"source{i}.parquet", columns=["entity_id", "country", "name_norm", "addr_norm"]).join(qid, on="entity_id", how="semi") for i in (2, 3)])
q = q.with_columns(france_eval.damage_addr(pl.col("addr_norm"), pl.col("entity_id")).alias("addr_norm"))
q = q.select("entity_id", "country", pl.concat_str([pl.lit("query: "), pl.col("name_norm"), pl.lit(" | "), pl.col("addr_norm")]).alias("text"))
d = embed.search(s1, lambda c: [q.filter(pl.col("country") == c)], embed.load_model(), top_k=EVAL_K)
d.write_parquet(NORM / "eval_dense_damaged.parquet")
print("eval_dense_damaged written", d.height)
