"""Structured pair features that expose WHAT differs between a record and an S1 candidate.

Why: the false positives of the ranker are near-copy decoys that differ in one detail (unit number `B-10` vs `B-9`,
a legal form, an extra name word). Fuzzy similarity hides such details and the old `num_eq` looked at the first number
only. These features are fed to LightGBM as separate columns so it can learn how much each kind of difference costs
(the penalty for a legal-form conflict, for example, is learned from data; it is not hand-set).

Per entity (entity_lists):  nums, units, tok_rare, tok_gen, legal_code, state_code
Per pair   (add_pair_features): counts / conflicts / equality flags of the above.

The vocabularies (frequent legal forms, generic name words) are built once from the S1 training file and stored in
models/struct_vocab.json so that train and test use exactly the same lists. The state vocabulary comes from the country
tables in normalize.py; codes of countries that never occur in training (France) simply get no training rows.
"""
import json
import os

import polars as pl

import normalize as N
from block import NORM, ROOT

VOCAB_PATH = ROOT / "models" / "struct_vocab.json"
N_LEGAL = 60      # most frequent legal forms of S1 that get their own code
N_GENERIC = 300   # most frequent (= generic) name words of S1: "group", "services", "holdings", ...

# French legal form -> the known (training) form whose code it borrows: limited-liability / single-member / simplified joint-stock /
# single-member simplified / public company / civil real-estate / general partnership / sole trader
FR_LEGAL = {"sarl": "llc", "eurl": "ltd", "sas": "inc", "sasu": "corp", "sa": "co", "sci": "llp", "snc": "lp", "ei": "pc"}
CATEGORICAL = ["legal_code_q", "legal_code_s", "state_code_q", "state_code_s"]
CONFLICT_FLAGS = ["num_conflict", "unit_conflict", "legal_conflict", "state_conflict"]  # constrained: conflict never raises p
STRUCT_FEATURES = [
    # address numbers (all numbers, leading zeros ignored) and letter+digit unit tokens ("b10", "4100b")
    "num_q_n", "num_s_n", "num_common", "num_only_q", "num_only_s", "num_jaccard", "num_equal", "num_subset",
    "num_conflict", "num_absdiff", "unit_common", "unit_only_q", "unit_only_s", "unit_conflict",
    # name words: rare words shared / only on one side, generic words only on one side
    "rare_common", "rare_only_q", "rare_only_s", "gen_only_q", "gen_only_s",
    # legal-form suffix
    "legal_same", "legal_conflict", "legal_missing_one", "legal_code_q", "legal_code_s",
    # states, separately for record and candidate, plus the pair verdict
    "state_code_q", "state_code_s", "state_conflict",
]


def state_vocab() -> dict[str, int]:
    codes = sorted({c for states, cs in N.COUNTRY_STATES.values() for c in list(cs) + list(states.values())})
    return {c: i + 1 for i, c in enumerate(codes)}  # 0 = unknown / missing


def load_vocab() -> dict:
    if VOCAB_PATH.exists():
        return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    s1 = pl.read_parquet(NORM / "source1.parquet", columns=["name_core", "legal_form"])
    legal = (s1.filter(pl.col("legal_form") != "")["legal_form"].value_counts()
               .sort("count", descending=True).head(N_LEGAL)["legal_form"].to_list())
    generic = (s1.select(pl.col("name_core").str.split(" ").alias("t")).explode("t")
                 .filter(pl.col("t").str.len_chars() >= 2).group_by("t").len()
                 .sort("len", descending=True).head(N_GENERIC)["t"].to_list())
    vocab = {"legal": legal, "generic": generic}
    VOCAB_PATH.parent.mkdir(exist_ok=True)
    VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return vocab


FR_VOCAB_PATH = ROOT / "models" / "struct_vocab_fr.json"
# French names use a small vocabulary: the top 100 words cover 63% of French S1 name tokens, in the range of the training list's coverage
# (US 43%, India 57.5%); top 300 would cover 73% and blunt the rare-word decoy signal. The swap words seen in unsure French pairs
# (societe, etablissements, compagnie, collectif, section, groupe, culturelle, association, developpement, services, publique) are all inside.
N_GENERIC_FR = 100


def fr_generic() -> list:
    """The generic (= frequent) name words of French S1 entities, by the same rule as the training vocabulary (top N_GENERIC words of S1 names).
    France has no training data, so its generic words were all counted as rare, distinctive words (a Societe <-> Etablissement swap looked like
    a different business). Built once from the S1 file of the split that holds French entities (test) and stored next to struct_vocab.json."""
    if FR_VOCAB_PATH.exists():
        return json.loads(FR_VOCAB_PATH.read_text(encoding="utf-8"))
    for f in ("test_source1.parquet", "source1.parquet"):
        p = NORM / f
        if p.exists():
            s1 = pl.read_parquet(p, columns=["country", "name_core"]).filter(pl.col("country") == "France")
            if s1.height:
                break
    else:
        return []
    words = (s1.select(pl.col("name_core").str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= 2)
               .group_by("t").len().sort("len", descending=True).head(N_GENERIC_FR)["t"].to_list())
    FR_VOCAB_PATH.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return words


def entity_lists(t: pl.DataFrame) -> pl.DataFrame:
    """Adds the per-entity list / code columns. Needs addr_latin, name_core, legal_form, state (and country for the French generic words)."""
    v = load_vocab()
    legal_idx = {x: i + 1 for i, x in enumerate(v["legal"])}
    # French legal forms never occur in training, so they all got code 0 ("no legal form"), which costs B2 about 0.006 on held-out data
    # (US / India pairs with their codes set to 0). A consistent mapping to distinct known codes recovers almost all of it (0.9864 vs 0.9869).
    # Each French form gets its own code of a close-ish known form; the strings never appear in US / India data, so nothing changes there.
    if os.environ.get("ER_FR_LEGAL", "1") == "1":
        for fr, known in FR_LEGAL.items():
            if known in legal_idx and fr not in legal_idx:
                legal_idx[fr] = legal_idx[known]
    generic = v["generic"]
    a = pl.col("addr_latin").fill_null("").str.to_lowercase()
    nums = (a.str.extract_all(r"\d+").list.eval(pl.element().cast(pl.Int64, strict=False))  # int cast drops leading zeros
             .list.drop_nulls().list.unique().list.sort())
    units = (a.str.extract_all(r"\b[a-z]{1,2}[\s\-/]?\d+[a-z]?\b|\b\d+[a-z]{1,2}\b")
              .list.eval(pl.element().str.replace_all(r"[^a-z0-9]", "")).list.unique().list.sort())
    toks = (pl.col("name_core").fill_null("").str.to_lowercase().str.split(" ")
              .list.eval(pl.element().filter(pl.element().str.len_chars() > 0)).list.unique())
    t = t.with_columns(
        nums.alias("nums"), units.alias("units"), toks.alias("_toks"),
        pl.col("legal_form").fill_null("").replace_strict(legal_idx, default=0, return_dtype=pl.Int16).alias("legal_code"),
        pl.col("state").fill_null("").replace_strict(state_vocab(), default=0, return_dtype=pl.Int16).alias("state_code"))
    if "country" in t.columns and os.environ.get("ER_FR_GENERIC", "1") == "1" and (t["country"] == "France").any():
        gen_fr = sorted(set(generic) | set(fr_generic()))   # French rows: the training list plus the French one; other rows unchanged
        fr = pl.col("country") == "France"
        return t.with_columns(
            pl.when(fr).then(pl.col("_toks").list.eval(pl.element().filter(pl.element().is_in(gen_fr))))
              .otherwise(pl.col("_toks").list.eval(pl.element().filter(pl.element().is_in(generic)))).alias("tok_gen"),
            pl.when(fr).then(pl.col("_toks").list.eval(pl.element().filter(~pl.element().is_in(gen_fr))))
              .otherwise(pl.col("_toks").list.eval(pl.element().filter(~pl.element().is_in(generic)))).alias("tok_rare")).drop("_toks")
    return t.with_columns(
        pl.col("_toks").list.eval(pl.element().filter(pl.element().is_in(generic))).alias("tok_gen"),
        pl.col("_toks").list.eval(pl.element().filter(~pl.element().is_in(generic))).alias("tok_rare")).drop("_toks")


def add_pair_features(c: pl.DataFrame) -> pl.DataFrame:
    """c has the q_* / s_* copies of entity_lists' columns (and q_legal_form / s_legal_form / q_state / s_state)."""
    i8 = pl.Int8
    qn, sn = pl.col("q_nums"), pl.col("s_nums")
    qu, su = pl.col("q_units"), pl.col("s_units")
    both_n = (qn.list.len() > 0) & (sn.list.len() > 0)
    both_u = (qu.list.len() > 0) & (su.list.len() > 0)
    n_int, n_only_q, n_only_s = qn.list.set_intersection(sn).list.len(), qn.list.set_difference(sn).list.len(), sn.list.set_difference(qn).list.len()
    u_int = qu.list.set_intersection(su).list.len()
    c = c.with_columns(
        qn.list.len().cast(i8).alias("num_q_n"), sn.list.len().cast(i8).alias("num_s_n"),
        n_int.cast(i8).alias("num_common"), n_only_q.cast(i8).alias("num_only_q"), n_only_s.cast(i8).alias("num_only_s"),
        (n_int / qn.list.set_union(sn).list.len().clip(1)).cast(pl.Float32).alias("num_jaccard"),
        (both_n & (n_only_q == 0) & (n_only_s == 0)).cast(i8).alias("num_equal"),
        (both_n & ((n_only_q == 0) | (n_only_s == 0)) & ~((n_only_q == 0) & (n_only_s == 0))).cast(i8).alias("num_subset"),
        (both_n & (n_int == 0)).cast(i8).alias("num_conflict"),
        # size of the smallest-number gap between what only one side has (349 vs 347: small gap = a tweaked decoy); log scale, -1 = n/a
        (qn.list.set_difference(sn).list.first() - sn.list.set_difference(qn).list.first()).abs().cast(pl.Float64).log1p().cast(pl.Float32).fill_null(-1.0).alias("num_absdiff"),
        u_int.cast(i8).alias("unit_common"),
        qu.list.set_difference(su).list.len().cast(i8).alias("unit_only_q"),
        su.list.set_difference(qu).list.len().cast(i8).alias("unit_only_s"),
        (both_u & (u_int == 0)).cast(i8).alias("unit_conflict"),
        pl.col("q_tok_rare").list.set_intersection(pl.col("s_tok_rare")).list.len().cast(i8).alias("rare_common"),
        pl.col("q_tok_rare").list.set_difference(pl.col("s_tok_rare")).list.len().cast(i8).alias("rare_only_q"),
        pl.col("s_tok_rare").list.set_difference(pl.col("q_tok_rare")).list.len().cast(i8).alias("rare_only_s"),
        pl.col("q_tok_gen").list.set_difference(pl.col("s_tok_gen")).list.len().cast(i8).alias("gen_only_q"),
        pl.col("s_tok_gen").list.set_difference(pl.col("q_tok_gen")).list.len().cast(i8).alias("gen_only_s"),
    )
    ql, sl = pl.col("q_legal_form").fill_null(""), pl.col("s_legal_form").fill_null("")
    both_l = (ql != "") & (sl != "")
    qs, ss = pl.col("q_state_code"), pl.col("s_state_code")
    return c.with_columns(
        (both_l & (ql == sl)).cast(i8).alias("legal_same"),
        (both_l & (ql != sl)).cast(i8).alias("legal_conflict"),
        (((ql == "") | (sl == "")) & ~((ql == "") & (sl == ""))).cast(i8).alias("legal_missing_one"),
        pl.col("q_legal_code").alias("legal_code_q"), pl.col("s_legal_code").alias("legal_code_s"),
        qs.alias("state_code_q"), ss.alias("state_code_s"),
        ((qs > 0) & (ss > 0) & (qs != ss)).cast(i8).alias("state_conflict"),
    )
