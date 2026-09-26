"""Run the whole entity-resolution pipeline, in order, from raw data to a validated submission.

  python main.py                        full run; steps whose outputs already exist are skipped (resumable)
  python main.py --list                 show every step and whether it is done
  python main.py --dry-run              print what would run (commands + environment), change nothing, check nothing
  python main.py --from ranker_fit_a    start at a step        --to write_submission   stop after a step
  python main.py --only block_test,embed_search_test
  python main.py --force                re-run steps even when their outputs exist
  python main.py --retrain ce,rankers,noaddr,rerank,pass2     train these components again in this run (see below)
  python main.py --skip-ce              no cross-encoder: ranker A only (shorter)
  python main.py --no-rerank            no name / address rerankers (model C)     --no-pass2   no sibling pass 2
  python main.py --skip-na              old single pipeline (name-only extras, one ranker for every record; no reranker / pass 2)
  python main.py --dataset /data/dataset --root /data/er_work     (or env ER_DATASET / ER_ROOT)
  --eval-frac 0.05 --train-frac 0.10     sample sizes (raise them only for a miniature test dataset)

Two pipelines (default):
  records WITH an address:    ranker A (structured features, hard-example weighting)
                              -> cross-encoder on the uncertain band -> ranker B (A + cross-encoder features)
                              -> pretrained name and address rerankers (listwise / contrastive fine-tuning, hard negatives) -> ranker C (B + both scores)
                              all LightGBMs are fitted on address records only; A, B or C is chosen on the held-out half no threshold was tuned on
                              (each must beat the previous best by more than 0.0005).
  records WITHOUT an address: char 3-gram TF-IDF blocking (blocking_noaddr.py) -> its own small LightGBM (noaddr.py), NO cross-encoder / reranker.
Then: joint decision rule -> score the test set -> patch the no-address rows -> (pass 2: the same pipeline over the TRAINING records = mirror,
sibling-support model fitted there, used only if it beats pass 1 by more than 0.0005 on the report half) -> write matching_results.tsv /
candidate_pairs.tsv -> report + deliverables -> the organisers' validator.

--retrain <components>: embed (e5 embedder), ce (cross-encoder), rankers (LightGBM A), rerank (name / address rerankers), noaddr (specialist), pass2, or all.
The weights of those components AND everything computed from them are moved to <root>/_replaced/<time>/ before the run, so their steps train again;
a dependent component is added automatically. Everything else in the work folder (normalized data, blocking / dense candidates, other models) is reused.

Every step is a separate process (the polars-heavy and the GPU-heavy stages do not share one), and its log is
written to logs/<step>.log. Thread counts and batch sizes are set from the machine (CPU, RAM, GPU memory) unless the
ER_* variables are already set. ER_CHUNK must not change between `block_test` and `score_a`: main.py checks reused test blocking against it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CODE = Path(__file__).resolve().parent
DEFAULT_DATASET = r"C:\Users\Lenovo\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
MARGIN = 0.0005     # a model / pass 2 must beat the previous best by this much on the half no threshold was tuned on


# ------------------------------------------------------------------------------------------ environment
def gpu_memory_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0])
    except Exception:
        return 0


def autoconfigure() -> dict:
    """Machine-dependent settings; anything already in the environment wins."""
    cpus = os.cpu_count() or 8
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**30
    except Exception:
        ram = 16.0
    gpu = gpu_memory_mib()
    cfg = {"ER_WORKERS": str(max(2, min(cpus - 2, 32))), "ER_THREADS": str(cpus),
           "ER_CHUNK": "200000" if ram >= 120 else "100000" if ram >= 60 else "50000"}
    if gpu >= 20000:
        cfg.update(ER_SEARCH_BATCH="1024", ER_ENCODE_BATCH="2048", ER_MINI_BATCH="512", ER_TRAIN_BATCH="1024",
                   ER_CE_BATCH="128", ER_CE_SCORE_BATCH="1024")
    for k, v in cfg.items():
        os.environ.setdefault(k, v)
    print(f"machine: {cpus} CPUs, {ram:.0f} GB RAM, GPU {gpu / 1024:.0f} GB   ->   " +
          ", ".join(f"{k}={os.environ[k]}" for k in cfg), flush=True)
    return cfg


def preflight(dataset: Path) -> list[str]:
    problems = []
    for sub in ("train", "test"):
        if not (dataset / sub).is_dir():
            problems.append(f"dataset folder missing: {dataset / sub}   (pass --dataset or set ER_DATASET)")
    import importlib.util
    for mod in ("polars", "numpy", "lightgbm", "rapidfuzz", "ftfy", "unidecode", "regex", "torch", "sentence_transformers", "transformers", "datasets", "sklearn"):
        if importlib.util.find_spec(mod) is None:
            problems.append(f"python package missing: {mod}   (pip install -r requirements.txt)")
    if gpu_memory_mib() == 0:
        problems.append("no NVIDIA GPU found (nvidia-smi): embedding fine-tuning / search / the cross-encoders need one")
    return problems


def reuse_problems(root: Path) -> list[str]:
    """Files reused from an earlier run must fit THIS run's settings; a mismatch would only show up hours later."""
    out = []
    N = root / "normalized"
    sp = N / "cand" / "test_sparse"
    srcs = [N / f"test_source{i}.parquet" for i in (2, 3)]
    if (sp / "_DONE").exists() and all(p.exists() for p in srcs):
        import polars as pl
        n = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in srcs)
        chunk = int(os.environ["ER_CHUNK"])
        need, have = -(-n // chunk), len(list(sp.glob("part*.parquet")))
        if have != need:
            out.append(f"the reused test blocking has {have} part files but ER_CHUNK={chunk} needs {need}: run with ER_CHUNK={-(-n // max(have, 1))} "
                       "(the value of the earlier run) or delete normalized/cand/test_sparse and test_dense so they are made again")
    return out


# ------------------------------------------------------------------------------------------ steps
@dataclass
class Step:
    name: str
    desc: str
    cmds: list = field(default_factory=list)          # each: argv after `python -u`
    env: dict = field(default_factory=dict)
    done: Callable[[], bool] = lambda: False
    when: Callable[[], bool] = lambda: True           # decided at run time (e.g. depends on the chosen model)
    run: Callable[[], None] | None = None             # in-process step instead of commands
    env_fn: Callable[[], dict] | None = None          # environment that is only known at run time (the chosen model)


def held(path: Path, key: str = "early-stop half"):
    """Expected official score of a decision file's held-out half that no threshold was tuned on (None when missing)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["held"][key]["official"]
    except Exception:
        return None


def build_steps(root: Path, dataset: Path, skip_ce: bool, eval_frac: float = 0.05, train_frac: float = 0.10, skip_na: bool = False,
                rerank: bool = True, pass2: bool = True) -> list[Step]:
    N, M = root / "normalized", root / "models"
    MIR = root / "fulltrain"                       # mirror folder: the TRAINING records play the test set (pass 2)
    MN, MM = MIR / "normalized", MIR / "models"
    exists = lambda *ps: (lambda: all(Path(p).exists() for p in ps))
    rerank = rerank and not skip_ce and not skip_na
    pass2 = pass2 and not skip_na
    ce = lambda: not skip_ce
    na = lambda: not skip_na
    # separate pipelines: the address models / cross-encoders never see records without an address (and skip the old name-only extras)
    AO = {} if skip_na else {"ER_ADDR_ONLY": "1", "ER_CE_ADDR_ONLY": "1", "ER_EXTRAS": "0"}
    ENVS = {   # the three address models: features tag, model file, decision file, score-part folder
        "A": {"ER_FEAT_TAG": "_s", "ER_MODEL": "ranker_a.txt", "ER_DECISION": "decision_a.json", **AO},
        "B": {"ER_CE": "1", "ER_FEAT_TAG": "_ce", "ER_MODEL": "ranker_b.txt", "ER_DECISION": "decision_b.json", **AO},
        "C": {"ER_CE": "1", "ER_CE2_TAG": "_nm", "ER_CE3_TAG": "_ad", "ER_FEAT_TAG": "_c", "ER_MODEL": "ranker_c.txt", "ER_DECISION": "decision_c.json", **AO},
    }
    A, B, C = ENVS["A"], ENVS["B"], ENVS["C"]
    PRED = {"A": "pred_a", "B": "pred_b", "C": "pred_c"}
    CE = {"ER_FEAT_TAG": "_s", "ER_STAGE1": "ranker_a.txt", **AO}
    # pretrained reranker (default BAAI/bge-reranker-v2-m3, Apache-2.0), listwise / contrastive fine-tuning on the record's owner vs its hard negatives
    RR = {**CE, "ER_CE_BASE": os.environ.get("RERANK_BASE", "BAAI/bge-reranker-v2-m3"), "ER_CE_DTYPE": "bf16", "ER_CE_LR": "2e-5", "ER_CE_LOSS": "listwise",
          "ER_CE_NEG": "6", "ER_CE_NEG_MINP": "0", "ER_CE_BATCH": "32", "ER_CE_ACCUM": "2", "ER_CE_SCORE_BATCH": os.environ.get("RERANK_SCORE_BATCH", "512"),
          "ER_CE_SAMPLE": os.environ.get("RERANK_SAMPLE", "0.1"), "ER_CE_EPOCHS": os.environ.get("RERANK_EPOCHS", "1")}
    RR_TRAIN = {"name": {**RR, "ER_CE_TAG": "_rr", "ER_CE_TEXT": "name", "ER_CE_MAXLEN": "64", "ER_CE_MODEL_DIR": "ce_rr_name"},
                "addr": {**RR, "ER_CE_TAG": "_rr", "ER_CE_TEXT": "addr", "ER_CE_MAXLEN": "96", "ER_CE_MODEL_DIR": "ce_rr_addr"}}
    RR_SCORE = {"name": {**RR_TRAIN["name"], "ER_CE_TAG": "_nm"}, "addr": {**RR_TRAIN["addr"], "ER_CE_TAG": "_ad"}}
    SUFFIX = {"name": "nm", "addr": "ad"}

    def final() -> dict:
        p = M / "final.json"
        return json.loads(p.read_text()) if p.exists() else {"choice": "A"}

    choice = lambda: final().get("choice", "A")
    joint_ce_needed = lambda: choice() in ("B", "C")
    rr_needed = lambda: choice() == "C"
    gate = lambda: (M / "pass2_gate.json").exists() and json.loads((M / "pass2_gate.json").read_text()).get("use") is True
    enabled = ["A"] + ([] if skip_ce else ["B"]) + (["C"] if rerank else [])

    def choose():
        """A, then B, then C: each must beat the best so far by MARGIN on the held-out half that no threshold was tuned on."""
        res = {l: held(M / ENVS[l]["ER_DECISION"]) for l in enabled}
        best = "A"
        for l in enabled[1:]:
            if res[l] is not None and res[best] is not None and res[l] > res[best] + MARGIN:
                best = l
        out = {"choice": best, **{f"official_{l.lower()}": res[l] for l in enabled}}
        (M / "final.json").write_text(json.dumps(out))
        print("held-out expected official score: " + "   ".join(f"{l} {res[l] if res[l] is None else round(res[l], 4)}" for l in enabled) + f"   ->  using model {best}", flush=True)

    def choose_done():
        if not (M / "final.json").exists():
            return False
        f = final()
        return all(f.get(f"official_{l.lower()}") is not None for l in enabled)

    def gate_pass2():
        """Copy the pass-2 model out of the mirror, decide whether it is used (beats pass 1 by MARGIN on the report half), free the mirror's big folders."""
        d = json.loads((MM / "decision_pass2.json").read_text())
        p2, p1 = d["held"]["report-half"]["official"], d["pass1_held"]["report-half"]["official"]
        use = p2 > p1 + MARGIN
        for f in ("pass2.txt", "decision_pass2.json"):
            shutil.copy2(MM / f, M / f)
        (M / "pass2_gate.json").write_text(json.dumps({"use": use, "pass1": p1, "pass2": p2, "margin": MARGIN}))
        print(f"pass 2: {p2:.4f} vs pass 1 {p1:.4f} on the report half -> {'USED' if use else 'NOT used'} (needs > +{MARGIN})", flush=True)
        if os.environ.get("KEEP_MIRROR", "0") != "1":   # about 20 GB: nothing after pass 2 needs the mirror's candidates or score parts
            for rel in ("cand",) + tuple(PRED.values()) + ("pred_na",):
                shutil.rmtree(MN / rel, ignore_errors=True)

    def final_pred() -> Path:
        return N / ("pred_pass2" if (pass2 and gate()) else "pred_na" if not skip_na else PRED[choice()])

    def submission_done():
        out = root / "output" / "matching_results.tsv"
        mark = final_pred() / "_DONE"
        deps = [p for p in (M / "final.json", M / "pass2_gate.json", mark) if p.exists()]
        return out.exists() and (M / "final.json").exists() and mark.exists() and out.stat().st_mtime > max(p.stat().st_mtime for p in deps)

    def address_env(extra: dict | None = None):
        """Environment of the CHOSEN address model (features tag, model file) for the no-address steps and the writer."""
        def f():
            l = choice()
            return {"ER_B2": ENVS[l]["ER_MODEL"], "ER_FEAT_TAG": ENVS[l]["ER_FEAT_TAG"], "ER_DECISION": "decision_na.json", **(extra or {})}
        return f

    def mirror(env: dict) -> dict:
        return {**env, "ER_ROOT": str(MIR)}

    validator = dataset.parent / "utils" / "validate_submission.py"
    steps = [
        Step("normalize_train", "clean the three training sources (mojibake, scripts, addresses, legal forms)",
             [["normalize.py", "train"]], done=exists(*[N / f"source{i}.parquet" for i in (1, 2, 3)])),
        Step("normalize_test", "clean the three test sources", [["normalize.py", "test"]],
             done=exists(*[N / f"test_source{i}.parquet" for i in (1, 2, 3)])),
        Step("block_eval", "sparse blocking of held-out queries (full S1 index) + held-out split",
             [["block.py", "eval", str(eval_frac)]], done=exists(N / "eval_sparse.parquet", N / "eval_queries.parquet")),
        Step("embed_finetune", "fine-tune multilingual-e5-small on (record, owner) pairs",
             [["embed.py", "train"]], done=exists(M / "e5_er" / "model.safetensors")),
        Step("embed_eval", "dense search of the held-out queries", [["embed.py", "eval"]], done=exists(N / "eval_dense.parquet")),
        Step("block_train", "sparse blocking for the ranker-training records", [["block.py", "train", str(train_frac)]],
             done=exists(N / "train_queries.parquet", N / "cand" / "train_sparse" / "_DONE")),
        Step("embed_search_train", "dense search for the ranker-training records", [["embed.py", "search", "train"]],
             done=exists(N / "cand" / "train_dense" / "_DONE")),
        Step("block_test", "sparse blocking of the test set", [["block.py", "test"]], done=exists(N / "cand" / "test_sparse" / "_DONE")),
        Step("embed_search_test", "dense search of the test set", [["embed.py", "search", "test"]],
             done=exists(N / "cand" / "test_dense" / "_DONE")),
    ]
    for split in ("eval", "train", "test"):
        if not skip_na:
            steps += [Step(f"na_block_{split}", f"char 3-gram TF-IDF candidates for the {split} records without an address",
                           [["blocking_noaddr.py", "build", split]], done=exists(N / "cand" / f"na_tfidf_{split}.parquet")),
                      Step(f"na_features_{split}", f"pair features of the no-address candidates ({split})", [["noaddr.py", "features", split]],
                           done=exists(N / f"feat_noaddr2_{split}.parquet"))]
            continue
        steps.append(Step(f"extras_{split}", f"name-only extra candidates for {split} records without an address",
                          [["extras.py", "sparse", split], ["extras.py", "dense", split], ["extras.py", "build", split]],
                          done=exists(N / "cand" / f"{split}_extra.parquet")))
    if not skip_na:   # the specialist learns from ALL no-address training records (about 310k), not only the 10% ranker-training sample
        steps += [
            Step("na_trainall_queries", "no-address training records for the specialist (held-out entities / evaluation queries excluded)",
                 [["noaddr.py", "trainall_queries"]], done=exists(N / "trainall_queries.parquet")),
            Step("na_trainall_sparse", "regular sparse candidates of those records", [["noaddr.py", "regular_sparse"]],
                 done=exists(N / "cand" / "trainall_sparse" / "_DONE")),
            Step("na_trainall_dense", "regular dense candidates of those records (e5)", [["noaddr.py", "regular_dense"]],
                 done=exists(N / "cand" / "trainall_dense" / "_DONE")),
            Step("na_block_trainall", "char 3-gram TF-IDF candidates of those records", [["blocking_noaddr.py", "build", "trainall"]],
                 done=exists(N / "cand" / "na_tfidf_trainall.parquet")),
            Step("na_features_trainall", "pair features of those candidates (with labels)", [["noaddr.py", "features", "trainall"]],
                 done=exists(N / "feat_noaddr2_trainall.parquet")),
        ]
    steps += [
        Step("features_a", "pair features (structured numbers / suffix / state features included), no cross-encoder",
             [["ranker.py", "features", "train"], ["ranker.py", "features", "eval"]], A,
             done=exists(N / "feat_train_s.parquet", N / "feat_eval_s.parquet")),
        Step("ranker_fit_a", "LightGBM ranker A: hard-example weighting, tuned decision rule", [["ranker.py", "fit"]], A,
             done=exists(M / "ranker_a.txt", M / "decision_a.json")),
        Step("ce_mine", "cross-encoder: mine hard positives and hard negatives", [["crossenc.py", "mine"]], CE,
             done=exists(N / "cepairs.parquet"), when=ce),
        Step("ce_train", "cross-encoder: fine-tune (3 epochs, as the laptop's round 2)", [["crossenc.py", "train"]],
             {**CE, "ER_CE_EPOCHS": os.environ.get("CE_EPOCHS", "3")},
             done=exists(M / "ce_er" / "model.safetensors"), when=ce),
        Step("ce_score_train", "cross-encoder: score the uncertain band of the training records", [["crossenc.py", "score", "train"]], CE,
             done=exists(N / "ce_train.parquet"), when=ce),
        Step("ce_score_eval", "cross-encoder: score the uncertain band of the held-out records", [["crossenc.py", "score", "eval"]], CE,
             done=exists(N / "ce_eval.parquet"), when=ce),
        Step("features_b", "pair features + cross-encoder scores", [["ranker.py", "features", "train"], ["ranker.py", "features", "eval"]], B,
             done=exists(N / "feat_train_ce.parquet", N / "feat_eval_ce.parquet"), when=ce),
        Step("ranker_fit_b", "LightGBM ranker B (with cross-encoder features)", [["ranker.py", "fit"]], B,
             done=exists(M / "ranker_b.txt", M / "decision_b.json"), when=ce),
    ]
    if rerank:
        steps.append(Step("rr_mine", "rerankers: mine the owner + up to 6 hard negatives per record (listwise groups)", [["crossenc.py", "mine"]],
                          {**RR, "ER_CE_TAG": "_rr"}, done=exists(N / "cepairs_rr.parquet")))
        for kind in ("name", "addr"):
            steps.append(Step(f"rr_train_{kind}", f"{kind} reranker: fine-tune the pretrained reranker with the listwise (contrastive) loss", [["crossenc.py", "train"]],
                              RR_TRAIN[kind], done=exists(M / RR_TRAIN[kind]["ER_CE_MODEL_DIR"] / "model.safetensors")))
        for split in ("train", "eval"):
            for kind in ("name", "addr"):
                steps.append(Step(f"rr_score_{split}_{kind}", f"{kind} reranker: score the uncertain band of the {split} records", [["crossenc.py", "score", split]],
                                  RR_SCORE[kind], done=exists(N / f"ce_{split}_{SUFFIX[kind]}.parquet")))
        steps += [
            Step("features_c", "pair features + cross-encoder + name / address reranker scores", [["ranker.py", "features", "train"], ["ranker.py", "features", "eval"]], C,
                 done=exists(N / "feat_train_c.parquet", N / "feat_eval_c.parquet")),
            Step("ranker_fit_c", "LightGBM ranker C (B + name and address reranker features)", [["ranker.py", "fit"]], C,
                 done=exists(M / "ranker_c.txt", M / "decision_c.json")),
        ]
    steps += [
        Step("choose_model", "pick A, B or C on the held-out half no threshold was tuned on", run=choose, done=choose_done),
        Step("na_fit", "no-address specialist LightGBM (no cross-encoder)", [["noaddr.py", "fit"]], done=exists(M / "noaddr_c.txt"), when=na,
             env_fn=address_env()),
        Step("na_joint", "joint decision rule: chosen address model + no-address specialist (held-out, two-half protocol)", [["noaddr.py", "joint"]],
             done=exists(M / "decision_na.json"), when=na, env_fn=address_env()),
        Step("score_a", "score every blocked test pair with ranker A (also gives stage-1 scores for the cross-encoder band)",
             [["predict.py", "score"]], {**A, "ER_PRED": str(N / "pred_a")}, done=exists(N / "pred_a" / "_DONE")),
        Step("ce_score_test", "cross-encoder: score the uncertain band of the test set", [["crossenc.py", "score", "test"]],
             {**CE, "ER_PRED": str(N / "pred_a")}, done=exists(N / "ce_test.parquet"), when=joint_ce_needed),
    ]
    if rerank:
        for kind in ("name", "addr"):
            steps.append(Step(f"rr_score_test_{kind}", f"{kind} reranker: score the uncertain band of the test set", [["crossenc.py", "score", "test"]],
                              {**RR_SCORE[kind], "ER_PRED": str(N / "pred_a")}, done=exists(N / f"ce_test_{SUFFIX[kind]}.parquet"), when=rr_needed))
    steps.append(Step("score_b", "score every blocked test pair with ranker B", [["predict.py", "score"]], {**B, "ER_PRED": str(N / "pred_b")},
                      done=exists(N / "pred_b" / "_DONE"), when=lambda: choice() == "B"))
    if rerank:
        steps.append(Step("score_c", "score every blocked test pair with ranker C", [["predict.py", "score"]], {**C, "ER_PRED": str(N / "pred_c")},
                          done=exists(N / "pred_c" / "_DONE"), when=lambda: choice() == "C"))
    steps.append(Step("na_apply", "replace the scores of the test records without an address by the specialist's", [["noaddr.py", "apply"]],
                      done=exists(N / "pred_na" / "_DONE"), when=na,
                      env_fn=lambda: {"ER_PRED_IN": str(N / PRED[choice()]), "ER_PRED_OUT": str(N / "pred_na")}))
    if pass2:
        steps += [
            Step("mirror_setup", "pass 2: mirror folder (hard links) in which the training records play the test set", [["fulltrain_setup.py"]],
                 {"ER_ROOT": str(root)}, done=exists(MN / "test_source1.parquet", MM / "noaddr_c.txt")),
            Step("mirror_block_test", "pass 2 mirror: sparse blocking of the training records", [["block.py", "test"]], {"ER_ROOT": str(MIR)},
                 done=exists(MN / "cand" / "test_sparse" / "_DONE")),
            Step("mirror_embed_search_test", "pass 2 mirror: dense search of the training records", [["embed.py", "search", "test"]], {"ER_ROOT": str(MIR)},
                 done=exists(MN / "cand" / "test_dense" / "_DONE")),
            Step("mirror_score_a", "pass 2 mirror: ranker A on every blocked pair", [["predict.py", "score"]], mirror({**A, "ER_PRED": str(MN / "pred_a")}),
                 done=exists(MN / "pred_a" / "_DONE")),
            Step("mirror_ce_score_test", "pass 2 mirror: cross-encoder band", [["crossenc.py", "score", "test"]], mirror({**CE, "ER_PRED": str(MN / "pred_a")}),
                 done=exists(MN / "ce_test.parquet"), when=joint_ce_needed),
        ]
        if rerank:
            for kind in ("name", "addr"):
                steps.append(Step(f"mirror_rr_score_test_{kind}", f"pass 2 mirror: {kind} reranker band", [["crossenc.py", "score", "test"]],
                                  mirror({**RR_SCORE[kind], "ER_PRED": str(MN / "pred_a")}), done=exists(MN / f"ce_test_{SUFFIX[kind]}.parquet"), when=rr_needed))
        steps.append(Step("mirror_score_b", "pass 2 mirror: ranker B", [["predict.py", "score"]], mirror({**B, "ER_PRED": str(MN / "pred_b")}),
                          done=exists(MN / "pred_b" / "_DONE"), when=lambda: choice() == "B"))
        if rerank:
            steps.append(Step("mirror_score_c", "pass 2 mirror: ranker C", [["predict.py", "score"]], mirror({**C, "ER_PRED": str(MN / "pred_c")}),
                              done=exists(MN / "pred_c" / "_DONE"), when=lambda: choice() == "C"))
        steps += [
            Step("mirror_na_block_test", "pass 2 mirror: TF-IDF candidates of the no-address training records", [["blocking_noaddr.py", "build", "test"]],
                 {"ER_ROOT": str(MIR)}, done=exists(MN / "cand" / "na_tfidf_test.parquet")),
            Step("mirror_na_features_test", "pass 2 mirror: features of the no-address candidates", [["noaddr.py", "features", "test"]], {"ER_ROOT": str(MIR)},
                 done=exists(MN / "feat_noaddr2_test.parquet")),
            Step("mirror_na_apply", "pass 2 mirror: specialist scores for the no-address records", [["noaddr.py", "apply"]], {"ER_ROOT": str(MIR)},
                 done=exists(MN / "pred_na" / "_DONE"), env_fn=lambda: {"ER_PRED_IN": str(MN / PRED[choice()]), "ER_PRED_OUT": str(MN / "pred_na")}),
            Step("pass2_fit", "pass 2: fit the sibling-support model on the mirror (out-of-sample records only) and compare with pass 1",
                 [["pass2.py", "fit"]], {"ER_ROOT": str(MIR), "ER_PASS1": str(MN / "pred_na"), "ER_INSAMPLE_DIR": str(N)},
                 done=exists(MM / "pass2.txt", MM / "decision_pass2.json")),
            Step("pass2_gate", "pass 2: use it only if it beats pass 1 on the report half", run=gate_pass2, done=exists(M / "pass2_gate.json")),
            Step("pass2_apply", "pass 2: re-score the real test set", [["pass2.py", "apply"]], {"ER_ROOT": str(root), "ER_PASS1": str(N / "pred_na")},
                 done=exists(N / "pred_pass2" / "_DONE"), when=gate),
        ]
    steps.append(Step("write_submission", "write matching_results.tsv and candidate_pairs.tsv with the tuned decision rule",
                      [["predict.py", "write"]], done=submission_done,
                      env_fn=lambda: ({"ER_DECISION": "decision_pass2.json", "ER_PRED": str(final_pred())} if (pass2 and gate()) else
                                      {"ER_DECISION": "decision_na.json", "ER_PRED": str(N / "pred_na")} if not skip_na else
                                      {"ER_DECISION": ENVS[choice()]["ER_DECISION"], "ER_PRED": str(N / PRED[choice()])})))
    steps.append(Step("report", "run report + deliverables/ (final files, held-out numbers, timeline, logs)", [["aws/report_full.py"]],
                      {"STATUS": "SUCCESS"}, done=lambda: False))
    steps.append(Step("validate", "organisers' validator on both output files",
                      [[str(validator), "--matching", str(root / "output" / "matching_results.tsv"),
                        "--candidate", str(root / "output" / "candidate_pairs.tsv"), "--test-dir", str(dataset / "test")]],
                      done=lambda: False, when=lambda: validator.exists()))
    return steps


# --retrain: the weights of these components are trained again in this run; everything derived from them is moved aside so its steps re-run.
# Everything else (normalized data, blocking and dense candidates, the other models) is reused as it is.
_RR_FILES = [f"normalized/ce_{s}_{t}.parquet" for s in ("train", "eval", "test") for t in ("nm", "ad")]
_C_FILES = ["normalized/feat_train_c.parquet", "normalized/feat_eval_c.parquet", "models/ranker_c.txt", "models/decision_c.json"]
RETRAIN_ARTIFACTS = {
    "embed":   ["models/e5_er", "normalized/eval_dense.parquet", "normalized/cand/train_dense", "normalized/cand/test_dense",
                "normalized/cand/eval_extra_dense.parquet", "normalized/cand/train_extra_dense.parquet", "normalized/cand/test_extra_dense.parquet",
                "normalized/feat_noaddr2_train.parquet", "normalized/feat_noaddr2_eval.parquet", "normalized/feat_noaddr2_test.parquet",
                "normalized/cand/trainall_dense", "normalized/feat_noaddr2_trainall.parquet"],
    "rankers": ["models/ranker_a.txt", "models/decision_a.json", "normalized/feat_train_s.parquet", "normalized/feat_eval_s.parquet"],
    "ce":      ["models/ce_er", "normalized/cepairs.parquet", "normalized/ce_train.parquet", "normalized/ce_eval.parquet", "normalized/ce_test.parquet",
                "normalized/feat_train_ce.parquet", "normalized/feat_eval_ce.parquet", "models/ranker_b.txt", "models/decision_b.json"] + _C_FILES,
    "rerank":  ["models/ce_rr_name", "models/ce_rr_addr", "normalized/cepairs_rr.parquet"] + _RR_FILES + _C_FILES,
    "noaddr":  ["models/noaddr_c.txt", "models/decision_na.json"],
    "pass2":   ["fulltrain", "models/pass2.txt", "models/decision_pass2.json", "models/pass2_gate.json", "normalized/pred_pass2"],
}
# a component also invalidates what is computed from it (embedder -> candidates -> all features; rankers -> the cross-encoder / reranker bands)
IMPLIES = {"embed": ["rankers", "ce", "rerank", "noaddr", "pass2"], "rankers": ["ce", "rerank", "pass2"], "ce": ["rerank", "pass2"],
           "rerank": ["pass2"], "noaddr": ["pass2"], "pass2": []}
# whatever is retrained: the model choice, every score part of the test set, the pass-2 mirror (it hard-links the models), and the old output
ALWAYS = ["models/final.json", "models/decision_na.json", "models/pass2_gate.json", "models/pass2.txt", "models/decision_pass2.json", "fulltrain",
          "normalized/pred_a", "normalized/pred_b", "normalized/pred_c", "normalized/pred_na", "normalized/pred_pass2", "output"]


def apply_retrain(root: Path, spec: str):
    """Move the artifacts of the chosen components (and what depends on them) to <root>/_replaced/<time>/. A marker makes a resumed run skip this."""
    marker = root / ".retrain_run"
    if marker.exists():
        print(f"--retrain: resuming the run started {marker.read_text().splitlines()[0]}", flush=True)
        return
    comps = set(c for c in spec.replace("all", ",".join(RETRAIN_ARTIFACTS)).split(",") if c)
    unknown = comps - set(RETRAIN_ARTIFACTS)
    if unknown:
        sys.exit(f"--retrain: unknown component(s) {sorted(unknown)}; choose from {sorted(RETRAIN_ARTIFACTS)} or 'all'")
    for c in list(comps):
        comps |= set(IMPLIES[c])
    dest = root / "_replaced" / time.strftime("%Y%m%d_%H%M%S")
    paths = sorted({p for c in comps for p in RETRAIN_ARTIFACTS[c]} | set(ALWAYS))
    moved = []
    for rel in paths:
        src = root / rel
        if src.exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest / rel))
            moved.append(rel)
    print(f"--retrain {sorted(comps)}: moved {len(moved)} earlier artifacts to {dest} ({', '.join(moved[:6])}{' ...' if len(moved) > 6 else ''}); their steps run again", flush=True)
    (root / "logs").mkdir(exist_ok=True)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + ",".join(sorted(comps)) + "\n")


def step_env(step: Step) -> dict:
    return {**step.env, **(step.env_fn() if step.env_fn else {})}


# ------------------------------------------------------------------------------------------ runner
def run_cmd(step: Step, argv: list, env_extra: dict, log_path: Path) -> int:
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONUTF8"] = "1"
    env.setdefault("ER_ROOT", str(env.get("ER_ROOT", CODE)))
    script = argv[0]
    cmd = [sys.executable, "-u", script if Path(script).is_absolute() else str(CODE / script), *argv[1:]]
    stop = threading.Event()
    t0 = time.time()

    def heartbeat():  # a line every 2 minutes with the last log line, so a long step is visibly alive
        while not stop.wait(120):
            try:
                last = log_path.read_text(errors="replace").strip().splitlines()[-1][:110]
            except Exception:
                last = ""
            print(f"    ... {step.name} running {time.time() - t0:.0f}s   {last}", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(argv)}\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    stop.set()
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("ER_DATASET") or DEFAULT_DATASET)
    ap.add_argument("--root", default=os.environ.get("ER_ROOT") or str(CODE), help="folder for normalized/, models/, output/, logs/")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--skip-ce", action="store_true"); ap.add_argument("--skip-na", action="store_true")
    ap.add_argument("--no-rerank", action="store_true"); ap.add_argument("--no-pass2", action="store_true")
    ap.add_argument("--retrain", default="", help="components trained again in this run (comma list of embed,ce,rankers,rerank,noaddr,pass2 or all); their old artifacts and everything derived from them are moved aside, the rest is reused")
    ap.add_argument("--from", dest="start"); ap.add_argument("--to", dest="stop"); ap.add_argument("--only")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--eval-frac", type=float, default=0.05, help="share of held-out records used as evaluation queries")
    ap.add_argument("--train-frac", type=float, default=0.10, help="share of records used to train the ranker")
    args = ap.parse_args()

    root, dataset = Path(args.root).resolve(), Path(args.dataset)
    os.environ["ER_ROOT"], os.environ["ER_DATASET"] = str(root), str(dataset)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    steps = build_steps(root, dataset, args.skip_ce, args.eval_frac, args.train_frac, args.skip_na, not args.no_rerank, not args.no_pass2)
    names = [s.name for s in steps]
    for opt in (args.start, args.stop):
        if opt and opt not in names:
            sys.exit(f"unknown step {opt!r}; steps: {', '.join(names)}")

    if args.list or args.dry_run:
        for s in steps:
            state = "done" if s.done() else "todo"
            print(f"{s.name:26s} [{state}]  {s.desc}")
            if args.dry_run:
                for c in s.cmds:
                    print(f"{'':28s}python {' '.join(c)}   {step_env(s) or ''}")
        return

    if args.retrain:
        apply_retrain(root, args.retrain)
    autoconfigure()
    if not args.no_preflight:
        problems = preflight(dataset) + reuse_problems(root)
        if problems:
            print("preflight problems:\n  - " + "\n  - ".join(problems))
            sys.exit(1)
    free = shutil.disk_usage(root).free / 2**30
    print(f"free disk: {free:.0f} GB  |  dataset: {dataset}  |  work folder: {root}", flush=True)

    only = set(args.only.split(",")) if args.only else None
    active = args.start is None
    summary = []
    t_all = time.time()
    for s in steps:
        if s.name == args.start:
            active = True
        if not active or (only and s.name not in only):
            continue
        try:
            if not s.when():
                print(f"-- {s.name}: not needed for this configuration / the chosen model", flush=True)
                summary.append((s.name, "not needed", 0)); continue
            if s.done() and not args.force:
                print(f"-- {s.name}: done already, skipping", flush=True)
                summary.append((s.name, "skipped", 0)); continue
            print(f"== {s.name}: {s.desc}", flush=True)
            t0 = time.time()
            if s.run:
                s.run()
            else:
                for argv in s.cmds:
                    log_path = root / "logs" / f"{s.name}.log"
                    rc = run_cmd(s, argv, step_env(s), log_path)
                    if rc != 0:
                        with open(root / "logs" / "timeline.tsv", "a", encoding="utf-8") as tl:
                            tl.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{s.name}\t{time.time() - t0:.0f}\tFAILED\n")
                        tail = "\n".join(log_path.read_text(errors="replace").strip().splitlines()[-15:])
                        print(f"!! {s.name} failed (exit {rc}); last log lines:\n{tail}\nfull log: {log_path}\n"
                              f"fix the cause and run `python main.py` again: finished steps are skipped.", flush=True)
                        sys.exit(rc)
            dt = time.time() - t0
            print(f"   {s.name} finished in {dt / 60:.1f} min", flush=True)
            summary.append((s.name, "ran", dt))
            with open(root / "logs" / "timeline.tsv", "a", encoding="utf-8") as tl:
                tl.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{s.name}\t{dt:.0f}\tok\n")
        except SystemExit:
            raise
        except Exception as e:
            print(f"!! {s.name} raised {type(e).__name__}: {e}", flush=True)
            sys.exit(1)
        if s.name == args.stop:
            break
    print("\nsummary")
    for name, state, dt in summary:
        print(f"  {name:26s} {state:11s} {dt / 60:7.1f} min" if state == "ran" else f"  {name:26s} {state}")
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
