"""Optional cross-encoder whose score is one more LightGBM feature (`ER_CE=1` switches the features on).

A cross-encoder reads the two records TOGETHER, so it can see the small differences the false positives hide
(`B-10` vs `B-9`, an extra word, another legal form) and, trained with hard POSITIVES as well, it also learns which
differences are only noise (leading zeros, reformatted units) and must not cost a true match.

  python crossenc.py mine               hard positives + hard negatives      -> normalized/cepairs<tag>.parquet
  python crossenc.py train              fine-tune multilingual-e5-small       -> models/ce_er/
  python crossenc.py score train|eval|test   score the uncertain band only    -> normalized/ce_<split>.parquet

Selection of what is scored (measured on the real test set: 7.0M pairs, about 50 min on an RTX 4050 at ~2,300 pairs/s):
  records whose best stage-1 probability lies in the uncertain band, plus records without an address (their extras);
  for each such record its top-K candidates by stage-1 probability.
Stage 1 is any saved ranker (ER_STAGE1, default the deployed one); the feature files it needs must exist.

Environment knobs: ER_STAGE1, ER_CE_BAND="0.05,0.99", ER_CE_TOPK=2, ER_CE_SAMPLE (fraction of train records to mine),
ER_CE_STEPS (stop training early: smoke test), ER_CE_MODEL_DIR, ER_CE_MAX_PAIRS (scoring cap: smoke test).
Reranker variants (a pretrained reranker as ER_CE_BASE, e.g. BAAI/bge-reranker-v2-m3):
  ER_CE_TEXT=joint|name|addr   what the model reads: name | address (default), the name only, or the address only -> separate name / address rerankers
  ER_CE_LOSS=bce|listwise      bce: one pair at a time (default); listwise: softmax over one record's candidates (its true owner + hard negatives),
                               the contrastive objective: the owner must outrank the look-alikes
  ER_CE_NEG=2                  hard negatives mined per record (use 6 for listwise) above the stage-1 probability ER_CE_NEG_MINP=0.05 (0 = always the top ones)
  ER_CE_GROUP=8                candidates per listwise group at most
  ER_CE_MAXLEN=128             token limit (name-only texts need far less)
Nothing here is trained by the pipeline until `train` is run explicitly.
"""
import math
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from block import NORM, ROOT, ground_truth
from ranker import CE_TAG, FEAT_TAG, load_extras

BASE_MODEL = os.environ.get("ER_CE_BASE", "intfloat/multilingual-e5-small")   # default: MIT licence, 118M parameters
# larger options (Apache-2.0 / MIT, all <= 8B): BAAI/bge-reranker-v2-m3 (568M), intfloat/multilingual-e5-large (560M), microsoft/mdeberta-v3-base (MIT)
TRUST = os.environ.get("ER_CE_TRUST_REMOTE_CODE", "0") == "1"   # gte-multilingual-reranker-base needs its own modelling code
CE_DIR = ROOT / "models" / os.environ.get("ER_CE_MODEL_DIR", "ce_er")
STAGE1 = ROOT / "models" / os.environ.get("ER_STAGE1", "ranker_final_backup.txt")
BAND = tuple(float(x) for x in os.environ.get("ER_CE_BAND", "0.05,0.99").split(","))
TOPK = int(os.environ.get("ER_CE_TOPK", 2))
MAX_LEN = int(os.environ.get("ER_CE_MAXLEN", 128))
TEXT = os.environ.get("ER_CE_TEXT", "joint")
HIDE_STATE = float(os.environ.get("ER_CE_HIDE_STATE", 0))   # share of entities whose trailing state code is removed from the text (1 = all, for the dependence test; 0.5 = training dropout)
LOSS = os.environ.get("ER_CE_LOSS", "bce")
NEG_PER_REC = int(os.environ.get("ER_CE_NEG", 2))
NEG_MINP = float(os.environ.get("ER_CE_NEG_MINP", 0.05))   # a wrong candidate counts as a hard negative above this stage-1 probability (0 = the top-NEG wrong ones of every record)
GROUP = int(os.environ.get("ER_CE_GROUP", 8))
ADDR_ONLY = os.environ.get("ER_CE_ADDR_ONLY", "0") == "1"   # separate pipelines: the cross-encoder only sees records WITH an address (~1.9M fewer test pairs)
BATCH = int(os.environ.get("ER_CE_BATCH", 64))              # 64 fits a 6 GB card, 128 a 24 GB card
SCORE_BATCH = int(os.environ.get("ER_CE_SCORE_BATCH", 256))
EPOCHS = int(os.environ.get("ER_CE_EPOCHS", 1))                # passes over the mined pairs
ACCUM = int(os.environ.get("ER_CE_ACCUM", 1))                  # micro-batches per optimiser step (effective batch = BATCH * ACCUM)
DTYPE = os.environ.get("ER_CE_DTYPE", "fp16")                  # "bf16" on Ampere+ cards (A10G, A100): no loss scaling needed, safer for large models


def stage1_probs(feat: pl.DataFrame) -> np.ndarray:
    m = lgb.Booster(model_file=str(STAGE1))
    return m.predict(feat.select(m.feature_name()).cast(pl.Float32).to_numpy())


def entity_text(prefix: str, ids: pl.DataFrame) -> pl.DataFrame:
    """entity_id, text = 'name | address' in the original script (normalized, states appended)."""
    d = pl.concat([pl.read_parquet(NORM / f"{prefix}source{i}.parquet", columns=["entity_id", "country", "name_norm", "addr_norm"]).join(ids, on="entity_id", how="semi")
                   for i in (1, 2, 3)])
    if os.environ.get("ER_FR_ADDR", "1") == "1":   # French rows only: canonical address text (ranker._fr_addr); no training data is French
        from ranker import _fr_addr
        d = d.with_columns(pl.when(pl.col("country") == "France").then(_fr_addr(pl.col("addr_norm"))).otherwise(pl.col("addr_norm")).alias("addr_norm"))
    d = d.drop("country")
    if HIDE_STATE > 0:   # hide the trailing state code (', tn' / ', az'): a country-specific token the model must not depend on
        d = d.join(pl.concat([pl.read_parquet(NORM / f"{prefix}source{i}.parquet", columns=["entity_id", "state"]).join(ids, on="entity_id", how="semi") for i in (1, 2, 3)]), on="entity_id", how="left")
        hide = (pl.col("entity_id").hash(seed=20260927) % 1000 < int(HIDE_STATE * 1000)) & (pl.col("state").fill_null("") != "") & pl.col("addr_norm").str.ends_with(", " + pl.col("state").fill_null(""))
        d = d.with_columns(pl.when(hide).then(pl.col("addr_norm").str.head(pl.col("addr_norm").str.len_chars() - pl.col("state").str.len_chars() - 2)).otherwise(pl.col("addr_norm")).alias("addr_norm")).drop("state")
    if TEXT == "name":
        return d.select("entity_id", pl.col("name_norm").fill_null("").alias("text"))
    if TEXT == "raw_name":   # the untouched business_name (case, punctuation, legal-form spelling kept): for records without an address
        r = pl.concat([pl.read_parquet(NORM / f"{prefix}source{i}.parquet", columns=["entity_id", "name_raw"]).join(ids, on="entity_id", how="semi") for i in (1, 2, 3)])
        return r.select("entity_id", pl.col("name_raw").fill_null("").alias("text"))
    if TEXT == "addr":
        return d.select("entity_id", pl.col("addr_norm").fill_null("").alias("text"))
    return d.select("entity_id", pl.concat_str([pl.col("name_norm"), pl.lit(" | "), pl.col("addr_norm")]).alias("text"))


def pair_texts(pairs: pl.DataFrame, prefix: str) -> pl.DataFrame:
    ids = pl.concat([pairs.select(pl.col("rec").alias("entity_id")), pairs.select(pl.col("s1").alias("entity_id"))]).unique()
    t = entity_text(prefix, ids)
    return (pairs.join(t.rename({"entity_id": "rec", "text": "text_a"}), on="rec", how="left")
                 .join(t.rename({"entity_id": "s1", "text": "text_b"}), on="s1", how="left"))


# ------------------------------------------------------------------------------------------------ mine
def mine():
    """Hard positives: true pairs stage 1 is unsure about. Hard negatives: the highest-scoring WRONG candidates of a
    record (twins, look-alikes). A few easy pairs of each class keep the model calibrated. Held-out S1 entities are
    never in the training candidates, so nothing here touches the evaluation set."""
    frac = float(os.environ.get("ER_CE_SAMPLE", 0.33))
    feat = pl.read_parquet(NORM / f"feat_train{FEAT_TAG}.parquet")
    feat = feat.filter(pl.col("rec").hash(seed=17) % 1000 < int(frac * 1000))
    if ADDR_ONLY:  # separate pipelines: the cross-encoder is trained on records WITH an address only
        feat = feat.filter(pl.col("q_has_addr") == 1)
    feat = feat.with_columns(pl.Series("p", stage1_probs(feat)))
    pos, neg = feat.filter(pl.col("label") == 1), feat.filter(pl.col("label") == 0)
    hard_pos = pos.filter(pl.col("p") < 0.90).with_columns(pl.lit(1).alias("hard"))
    if LOSS == "listwise":  # every group needs its true owner: all positives of the sampled records, each with its top-NEG wrong candidates
        easy_pos = pos.filter(pl.col("p") >= 0.90).with_columns(pl.lit(0).alias("hard"))
    else:
        easy_pos = pos.filter(pl.col("p") >= 0.90).sample(n=min(hard_pos.height, pos.height - hard_pos.height), seed=1).with_columns(pl.lit(0).alias("hard"))
    hard_neg = (neg.filter(pl.col("p") > NEG_MINP).sort("p", descending=True).group_by("rec", maintain_order=True).head(NEG_PER_REC)
                   .with_columns(pl.lit(1).alias("hard")))
    easy_neg = neg.filter(pl.col("p") <= 0.05).sample(n=min(hard_neg.height // 2 + 1, 200_000), seed=2).with_columns(pl.lit(0).alias("hard"))
    if LOSS == "listwise":  # a listwise group only uses the record's own candidates: random easy negatives of other records add nothing
        easy_neg = easy_neg.head(0)
    out = pl.concat([x.select("rec", "s1", "label", "hard") for x in (hard_pos, easy_pos, hard_neg, easy_neg)])
    out.write_parquet(NORM / f"cepairs{CE_TAG}.parquet")
    print(f"hard positives {hard_pos.height:,}  easy positives {easy_pos.height:,}  hard negatives {hard_neg.height:,}  "
          f"easy negatives {easy_neg.height:,}  -> {out.height:,} pairs", flush=True)


# ------------------------------------------------------------------------------------------------ train
def train():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    base_pairs = pl.read_parquet(NORM / f"cepairs{CE_TAG}.parquet")
    for extra in [t for t in os.environ.get("ER_CE_EXTRA_PAIRS", "").split(",") if t]:  # pair files of earlier rounds (tags), added to this round's
        base_pairs = pl.concat([base_pairs, pl.read_parquet(NORM / f"cepairs{extra}.parquet")]).unique(subset=["rec", "s1"], keep="first", maintain_order=True)
    print(f"training pairs: {base_pairs.height:,}  (extra pair files: {os.environ.get('ER_CE_EXTRA_PAIRS', '-')}; init from: {os.environ.get('ER_CE_INIT', 'base model')})", flush=True)
    pairs = pair_texts(base_pairs, "").drop_nulls().sample(fraction=1.0, shuffle=True, seed=0)
    if LOSS == "listwise":
        return train_listwise(pairs)
    eff = BATCH * ACCUM
    steps_per_epoch = math.ceil(pairs.height / eff)
    steps_total = steps_per_epoch * EPOCHS
    max_steps = int(os.environ.get("ER_CE_STEPS", steps_total))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST)
    init = os.environ.get("ER_CE_INIT")  # warm start from an earlier round (a directory under models/)
    model = AutoModelForSequenceClassification.from_pretrained(str(ROOT / "models" / init) if init else BASE_MODEL, num_labels=1, trust_remote_code=TRUST).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("ER_CE_LR", 3e-5)), weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * max_steps), max_steps)
    adt = torch.bfloat16 if DTYPE == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=adt == torch.float16)
    model.train()
    t0, run = time.time(), 0.0
    for step in range(max_steps):
        if step % steps_per_epoch == 0:  # a fresh shuffle for every epoch
            ep = pairs.sample(fraction=1.0, shuffle=True, seed=step // steps_per_epoch)
            a, b, y = ep["text_a"].to_list(), ep["text_b"].to_list(), ep["label"].to_numpy().astype(np.float32)
        opt.zero_grad(set_to_none=True)
        for k in range(ACCUM):
            i = (step % steps_per_epoch) * eff + k * BATCH
            if i >= len(y):
                break
            enc = tok(a[i:i + BATCH], b[i:i + BATCH], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=adt):
                logit = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[i:i + BATCH]).cuda())
            scaler.scale(loss / ACCUM).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0:
            print(f"  step {step}/{max_steps}  loss {run:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    CE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CE_DIR); tok.save_pretrained(CE_DIR)
    print("saved", CE_DIR, flush=True)


def train_listwise(pairs: pl.DataFrame):
    """Contrastive / listwise fine-tuning: for every record with a mined true owner and at least one hard negative, the model scores the group
    [owner, negative 1, ..., negative k] and is trained with a softmax cross-entropy that asks for the owner to get the top score."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    g = (pairs.sort("label", descending=True).group_by("rec", maintain_order=True)
              .agg(pl.col("text_a").first(), pl.col("text_b"), pl.col("label"))
              .filter((pl.col("label").list.first() == 1) & (pl.col("label").list.len() >= 2))
              .with_columns(pl.col("text_b").list.head(GROUP)))
    A, B = g["text_a"].to_list(), g["text_b"].to_list()
    n = len(A)
    per = max(1, BATCH // GROUP)                      # groups per micro-batch (about BATCH pairs)
    eff = per * ACCUM
    steps_per_epoch = math.ceil(n / eff)
    max_steps = int(os.environ.get("ER_CE_STEPS", steps_per_epoch * EPOCHS))
    print(f"listwise groups: {n:,} (one true owner + up to {GROUP - 1} hard negatives each); {steps_per_epoch} steps per epoch, {max_steps} steps", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST)
    init = os.environ.get("ER_CE_INIT")
    model = AutoModelForSequenceClassification.from_pretrained(str(ROOT / "models" / init) if init else BASE_MODEL, num_labels=1, trust_remote_code=TRUST).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("ER_CE_LR", 3e-5)), weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * max_steps), max_steps)
    adt = torch.bfloat16 if DTYPE == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=adt == torch.float16)
    rng = np.random.default_rng(0)
    model.train()
    t0, run, perm = time.time(), 0.0, None
    for step in range(max_steps):
        if step % steps_per_epoch == 0:
            perm = rng.permutation(n)
        opt.zero_grad(set_to_none=True)
        for k in range(ACCUM):
            idx = perm[(step % steps_per_epoch) * eff + k * per:(step % steps_per_epoch) * eff + (k + 1) * per]
            if len(idx) == 0:
                break
            ta, tb, sizes = [], [], []
            for i in idx:
                ta += [A[i]] * len(B[i]); tb += B[i]; sizes.append(len(B[i]))
            enc = tok(ta, tb, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=adt):
                logit = model(**enc).logits.squeeze(-1).float()
            loss, o = 0.0, 0
            for sz in sizes:  # the owner is entry 0 of every group
                loss = loss + F.cross_entropy(logit[o:o + sz].unsqueeze(0), torch.zeros(1, dtype=torch.long, device=logit.device))
                o += sz
            loss = loss / len(sizes)
            scaler.scale(loss / ACCUM).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step else float(loss)
        if step % 100 == 0:
            print(f"  step {step}/{max_steps}  listwise loss {run:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    CE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CE_DIR); tok.save_pretrained(CE_DIR)
    print("saved", CE_DIR, flush=True)


# ------------------------------------------------------------------------------------------------ score
def band_pairs(split: str) -> tuple[pl.DataFrame, str]:
    """(rec, s1) pairs of the uncertain band for a split, and the entity-file prefix.
    Band = records whose best stage-1 probability lies in BAND, plus records without an address; for each, the top-TOPK
    candidates by stage-1 probability (+ the name-only extras of the records without an address)."""
    lo, hi = BAND
    if split == "test":
        # stage-1 probabilities of the test pairs were written by `predict.py score`: (rec_i, s1_i, p) as integers.
        # Everything is decided on integers; string ids are attached only to the ~10M selected pairs.
        from predict import PRED, ids_table
        names = ids_table()["entity_id"]
        n_s1 = pl.scan_parquet(NORM / "test_source1.parquet").select(pl.len()).collect().item()
        has_addr = pl.concat([pl.scan_parquet(NORM / f"test_source{i}.parquet").select("has_addr") for i in (2, 3)]).collect()["has_addr"].to_numpy()
        d = pl.scan_parquet(PRED / "*.parquet").collect().sort("p", descending=True)
        best = d.group_by("rec_i", maintain_order=True).agg(pl.col("p").first().alias("p1"))
        addr = has_addr[best["rec_i"].to_numpy() - n_s1]  # S2/S3 records follow the S1 rows in the id table
        band = best.filter(pl.Series(((best["p1"].to_numpy() >= lo) & (best["p1"].to_numpy() < hi)) | (~addr & (not ADDR_ONLY)))).select("rec_i")
        top = (d.join(band, on="rec_i", how="semi").with_columns(pl.int_range(1, pl.len() + 1).over("rec_i").alias("r"))
                .filter(pl.col("r") <= TOPK))
        del d
        pairs = pl.DataFrame({"rec": names.gather(top["rec_i"].to_numpy()), "s1": names.gather(top["s1_i"].to_numpy())})
        band_ids = pl.DataFrame({"rec": names.gather(band["rec_i"].to_numpy())})
        prefix = "test_"
    else:
        feat = pl.read_parquet(NORM / f"feat_{split}{FEAT_TAG}.parquet")
        d = (feat.select("rec", "s1", "q_has_addr").rename({"q_has_addr": "has_addr"})
                 .with_columns(pl.Series("p", stage1_probs(feat)), pl.col("has_addr").cast(pl.Boolean)).sort("p", descending=True))
        del feat
        best = d.group_by("rec", maintain_order=True).agg(pl.col("p").first().alias("p1"), pl.col("has_addr").first())
        band_ids = best.filter(((pl.col("p1") >= lo) & (pl.col("p1") < hi)) | (~pl.col("has_addr") & (not ADDR_ONLY))).select("rec")
        pairs = (d.join(band_ids, on="rec", how="semi").with_columns(pl.int_range(1, pl.len() + 1).over("rec").alias("r"))
                  .filter(pl.col("r") <= TOPK).select("rec", "s1"))
        prefix = ""
    ex = None if ADDR_ONLY else load_extras(split)  # name-only extras of the records without an address are scored too
    if ex is not None:
        pairs = pl.concat([pairs, ex.select("rec", "s1").join(band_ids, on="rec", how="semi")]).unique()
    return pairs, prefix


def score(split: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    pairs, prefix = band_pairs(split) if not os.environ.get("ER_CE_PAIRS") else (pl.read_parquet(os.environ["ER_CE_PAIRS"]).select("rec", "s1"), "test_" if split == "test" else "")
    # ER_CE_PAIRS=<parquet of (rec, s1)>: score exactly these pairs (noaddr.py cascade) instead of the uncertain band
    cap = int(os.environ.get("ER_CE_MAX_PAIRS", 0))
    if cap:
        pairs = pairs.head(cap)
    print(f"{split}: {pairs.height:,} pairs in the uncertain band / without address to score ({time.time() - t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(CE_DIR, trust_remote_code=TRUST)
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR, trust_remote_code=TRUST).cuda().to(torch.bfloat16 if DTYPE == "bf16" else torch.float16).eval()
    parts, step = [], 500_000
    for lo in range(0, pairs.height, step):
        pt = pair_texts(pairs.slice(lo, step), prefix).with_columns(pl.col("text_a").fill_null(""), pl.col("text_b").fill_null(""))
        pt = pt.sort(pl.col("text_a").str.len_chars() + pl.col("text_b").str.len_chars())  # similar lengths per batch
        a, b, out = pt["text_a"].to_list(), pt["text_b"].to_list(), []
        with torch.no_grad():
            for i in range(0, len(a), SCORE_BATCH):
                enc = tok(a[i:i + SCORE_BATCH], b[i:i + SCORE_BATCH], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
                out.append(torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy())
        parts.append(pt.select("rec", "s1").with_columns(pl.Series("ce_score", np.concatenate(out), dtype=pl.Float32)))
        print(f"  scored {min(lo + step, pairs.height):,}/{pairs.height:,} ({time.time() - t0:.0f}s)", flush=True)
    pl.concat(parts).write_parquet(NORM / f"ce_{split}{CE_TAG}.parquet")


if __name__ == "__main__":
    {"mine": mine, "train": train, "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
