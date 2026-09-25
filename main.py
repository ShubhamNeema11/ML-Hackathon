"""Run the whole entity-resolution pipeline, in order, from raw data to a validated submission.

  python main.py                        full run; steps whose outputs already exist are skipped (resumable)
  python main.py --list                 show every step and whether it is done
  python main.py --dry-run              print what would run, change nothing
  python main.py --from ranker_fit_a    start at a step        --to write_submission   stop after a step
  python main.py --only block_test,embed_search_test
  python main.py --force                re-run steps even when their outputs exist
  python main.py --skip-ce              no cross-encoder: ranker A only (shorter)
  python main.py --dataset /data/dataset --root /data/er_work     (or env ER_DATASET / ER_ROOT)
  --eval-frac 0.05 --train-frac 0.10     sample sizes (raise them only for a miniature test dataset)

Order:  normalize -> sparse + dense blocking (held-out eval, ranker-train and test sets) -> name-only extras ->
        ranker A (structured features, hard-example weighting, tuned decision rule) ->
        cross-encoder (mine hard pairs, train, score the uncertain band) -> ranker B (A + cross-encoder features) ->
        choose A or B on the held-out entities -> score the test set -> write matching_results.tsv / candidate_pairs.tsv ->
        run the organisers' validator.

Every step is a separate process (the polars-heavy and the GPU-heavy stages do not share one), and its log is
written to logs/<step>.log. Thread counts and batch sizes are set from the machine (CPU, RAM, GPU memory) unless the
ER_* variables are already set. ER_CHUNK must not change between `block_test` and `score_a`; main.py sets it once.
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
    for mod in ("polars", "numpy", "lightgbm", "rapidfuzz", "ftfy", "unidecode", "regex", "torch", "sentence_transformers", "transformers", "datasets"):
        if importlib.util.find_spec(mod) is None:
            problems.append(f"python package missing: {mod}   (pip install -r requirements.txt)")
    if gpu_memory_mib() == 0:
        problems.append("no NVIDIA GPU found (nvidia-smi): embedding fine-tuning / search need one")
    return problems


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


def build_steps(root: Path, dataset: Path, skip_ce: bool, eval_frac: float = 0.05, train_frac: float = 0.10) -> list[Step]:
    N, M = root / "normalized", root / "models"
    exists = lambda *ps: (lambda: all(Path(p).exists() for p in ps))
    A = {"ER_FEAT_TAG": "_s", "ER_MODEL": "ranker_a.txt", "ER_DECISION": "decision_a.json"}
    B = {"ER_CE": "1", "ER_FEAT_TAG": "_ce", "ER_MODEL": "ranker_b.txt", "ER_DECISION": "decision_b.json"}
    CE = {"ER_FEAT_TAG": "_s", "ER_STAGE1": "ranker_a.txt"}
    final = lambda: json.loads((M / "final.json").read_text()) if (M / "final.json").exists() else {"choice": "A"}
    use_b = lambda: final().get("choice") == "B"
    ce = lambda: not skip_ce

    def choose():
        """Model B (A + cross-encoder features) must beat A on the held-out half that no threshold was tuned on."""
        a = json.loads((M / "decision_a.json").read_text())["held"]["early-stop half"]["official"]
        choice, b = "A", None
        if not skip_ce and (M / "decision_b.json").exists():
            b = json.loads((M / "decision_b.json").read_text())["held"]["early-stop half"]["official"]
            choice = "B" if b > a + 0.0005 else "A"   # a margin, so noise cannot flip the choice
        (M / "final.json").write_text(json.dumps({"choice": choice, "official_a": a, "official_b": b}))
        print(f"held-out expected official score: A {a:.4f}   B {b if b is None else round(b, 4)}   ->  using model {choice}", flush=True)

    def submission_done():
        out = root / "output" / "matching_results.tsv"
        pred = N / ("pred_b" if use_b() else "pred_a") / "_DONE"
        return out.exists() and (M / "final.json").exists() and out.stat().st_mtime > max((M / "final.json").stat().st_mtime, pred.stat().st_mtime if pred.exists() else 0)

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
        steps.append(Step(f"extras_{split}", f"name-only extra candidates for {split} records without an address",
                          [["extras.py", "sparse", split], ["extras.py", "dense", split], ["extras.py", "build", split]],
                          done=exists(N / "cand" / f"{split}_extra.parquet")))
    steps += [
        Step("features_a", "pair features (structured numbers / suffix / state features included), no cross-encoder",
             [["ranker.py", "features", "train"], ["ranker.py", "features", "eval"]], A,
             done=exists(N / "feat_train_s.parquet", N / "feat_eval_s.parquet")),
        Step("ranker_fit_a", "LightGBM ranker A: hard-example weighting, tuned decision rule", [["ranker.py", "fit"]], A,
             done=exists(M / "ranker_a.txt", M / "decision_a.json")),
        Step("ce_mine", "cross-encoder: mine hard positives and hard negatives", [["crossenc.py", "mine"]], CE,
             done=exists(N / "cepairs.parquet"), when=ce),
        Step("ce_train", "cross-encoder: fine-tune", [["crossenc.py", "train"]], CE,
             done=exists(M / "ce_er" / "model.safetensors"), when=ce),
        Step("ce_score_train", "cross-encoder: score the uncertain band of the training records", [["crossenc.py", "score", "train"]], CE,
             done=exists(N / "ce_train.parquet"), when=ce),
        Step("ce_score_eval", "cross-encoder: score the uncertain band of the held-out records", [["crossenc.py", "score", "eval"]], CE,
             done=exists(N / "ce_eval.parquet"), when=ce),
        Step("features_b", "pair features + cross-encoder scores", [["ranker.py", "features", "train"], ["ranker.py", "features", "eval"]], B,
             done=exists(N / "feat_train_ce.parquet", N / "feat_eval_ce.parquet"), when=ce),
        Step("ranker_fit_b", "LightGBM ranker B (with cross-encoder features)", [["ranker.py", "fit"]], B,
             done=exists(M / "ranker_b.txt", M / "decision_b.json"), when=ce),
        Step("choose_model", "pick A or B on the held-out half no threshold was tuned on", run=choose,
             done=lambda: (M / "final.json").exists() and (skip_ce or json.loads((M / "final.json").read_text()).get("official_b") is not None)),
        Step("score_a", "score every blocked test pair with ranker A (also gives stage-1 scores for the cross-encoder band)",
             [["predict.py", "score"]], {**A, "ER_PRED": str(N / "pred_a")}, done=exists(N / "pred_a" / "_DONE")),
        Step("ce_score_test", "cross-encoder: score the uncertain band of the test set", [["crossenc.py", "score", "test"]],
             {**CE, "ER_PRED": str(N / "pred_a")}, done=exists(N / "ce_test.parquet"), when=lambda: use_b()),
        Step("score_b", "score every blocked test pair with ranker B", [["predict.py", "score"]], {**B, "ER_PRED": str(N / "pred_b")},
             done=exists(N / "pred_b" / "_DONE"), when=lambda: use_b()),
        Step("write_submission", "write matching_results.tsv and candidate_pairs.tsv with the tuned decision rule",
             [["predict.py", "write"]], done=submission_done),
    ]
    steps.append(Step("validate", "organisers' validator on both output files",
                      [[str(validator), "--matching", str(root / "output" / "matching_results.tsv"),
                        "--candidate", str(root / "output" / "candidate_pairs.tsv"), "--test-dir", str(dataset / "test")]],
                      done=lambda: False, when=lambda: validator.exists()))
    return steps


def step_env(step: Step, root: Path, use_b: bool) -> dict:
    env = dict(step.env)
    if step.name == "write_submission":  # the winning model's decision rule and score parts (known only after choose_model)
        n = root / "normalized"
        env.update({"ER_DECISION": "decision_b.json" if use_b else "decision_a.json", "ER_PRED": str(n / ("pred_b" if use_b else "pred_a"))})
    return env


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
    ap.add_argument("--force", action="store_true"); ap.add_argument("--skip-ce", action="store_true")
    ap.add_argument("--from", dest="start"); ap.add_argument("--to", dest="stop"); ap.add_argument("--only")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--eval-frac", type=float, default=0.05, help="share of held-out records used as evaluation queries")
    ap.add_argument("--train-frac", type=float, default=0.10, help="share of records used to train the ranker")
    args = ap.parse_args()

    root, dataset = Path(args.root).resolve(), Path(args.dataset)
    os.environ["ER_ROOT"], os.environ["ER_DATASET"] = str(root), str(dataset)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    steps = build_steps(root, dataset, args.skip_ce, args.eval_frac, args.train_frac)
    names = [s.name for s in steps]
    for opt in (args.start, args.stop):
        if opt and opt not in names:
            sys.exit(f"unknown step {opt!r}; steps: {', '.join(names)}")

    if args.list or args.dry_run:
        for s in steps:
            state = "done" if s.done() else "todo"
            print(f"{s.name:20s} [{state}]  {s.desc}")
            if args.dry_run:
                for c in s.cmds:
                    print(f"{'':22s}python {' '.join(c)}   {step_env(s, root, False) or ''}")
        return

    autoconfigure()
    if not args.no_preflight:
        problems = preflight(dataset)
        if problems:
            print("preflight problems:\n  - " + "\n  - ".join(problems))
            sys.exit(1)
    free = shutil.disk_usage(root).free / 2**30
    print(f"free disk: {free:.0f} GB (a full run writes about 60 GB)  |  dataset: {dataset}  |  work folder: {root}", flush=True)

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
                print(f"-- {s.name}: not needed ({'cross-encoder skipped' if args.skip_ce else 'model A was chosen'})", flush=True)
                summary.append((s.name, "not needed", 0)); continue
            if s.done() and not args.force:
                print(f"-- {s.name}: done already, skipping", flush=True)
                summary.append((s.name, "skipped", 0)); continue
            print(f"== {s.name}: {s.desc}", flush=True)
            t0 = time.time()
            if s.run:
                s.run()
            else:
                use_b = (root / "models" / "final.json").exists() and json.loads((root / "models" / "final.json").read_text()).get("choice") == "B"
                for argv in s.cmds:
                    log_path = root / "logs" / f"{s.name}.log"
                    rc = run_cmd(s, argv, step_env(s, root, use_b), log_path)
                    if rc != 0:
                        tail = "\n".join(log_path.read_text(errors="replace").strip().splitlines()[-15:])
                        print(f"!! {s.name} failed (exit {rc}); last log lines:\n{tail}\nfull log: {log_path}\n"
                              f"fix the cause and run `python main.py` again: finished steps are skipped.", flush=True)
                        sys.exit(rc)
            dt = time.time() - t0
            print(f"   {s.name} finished in {dt / 60:.1f} min", flush=True)
            summary.append((s.name, "ran", dt))
        except SystemExit:
            raise
        except Exception as e:
            print(f"!! {s.name} raised {type(e).__name__}: {e}", flush=True)
            sys.exit(1)
        if s.name == args.stop:
            break
    print("\nsummary")
    for name, state, dt in summary:
        print(f"  {name:20s} {state:11s} {dt / 60:7.1f} min" if state == "ran" else f"  {name:20s} {state}")
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
