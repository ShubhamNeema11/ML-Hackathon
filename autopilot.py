"""Unattended finish: E2 result -> choose model -> cross-encoder on the test set -> ranker on the test set -> files -> validator.

Run through the supervisor:   bash run_autopilot.sh          (restarts this script if it dies; every step is resumable)
Dry run (plan + decision only): python autopilot.py --dry

Output: final_overnight/matching_results.tsv, final_overnight/candidate_pairs.tsv, final_overnight/READY.txt (the morning summary).
The previously validated hybrid submission in final_submission/ is never touched.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

CODE = Path(__file__).resolve().parent
NORM, MODELS, LOGS = CODE / "normalized", CODE / "models", CODE / "logs"
OUT = CODE / "final_overnight"
DATASET = Path(r"C:\Users\Lenovo\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset")
VALIDATOR = DATASET.parent / "utils" / "validate_submission.py"
MARGIN = 0.0005  # B2 must beat B1 by more than this on the held-out half no threshold was tuned on
RETRIES = 3

CONFIG = {
    "B1": dict(ce_tag="", ce_dir="ce_er", ranker="ranker_b.txt", decision="decision_b.json"),
    "B2": dict(ce_tag="_v2", ce_dir="ce_er2", ranker="ranker_b2.txt", decision="decision_b2.json"),
}


def say(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)


def keep_awake():
    """Ask Windows not to sleep while this process lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); reverts when it exits."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def held(decision_file: str) -> dict | None:
    p = MODELS / decision_file
    return json.loads(p.read_text())["held"] if p.exists() else None


def e2_state() -> str:
    """'done', 'failed' or 'running' for the E2 experiment chain (run_e2.sh)."""
    log = LOGS / "exp_e2.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    if "E2 DONE" in text:
        return "done"
    running = any("run_e2.sh" in " ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))
    return "running" if running else "failed"


def choose() -> str:
    f = MODELS / "final_overnight.json"
    if f.exists():
        return json.loads(f.read_text())["choice"]
    while (state := e2_state()) == "running":
        time.sleep(30)
    b1 = held("decision_b.json")
    b2 = held("decision_b2.json") if state == "done" else None
    choice, why = "B1", f"E2 {state}: keeping B1"
    if b2 and b1:
        e1, e2 = b1["early-stop half"]["official"], b2["early-stop half"]["official"]
        t1, t2 = b1["thr-half"]["official"], b2["thr-half"]["official"]
        if e2 > e1 + MARGIN and t2 > t1 - 0.002:
            choice = "B2"
        why = (f"held-out expected official, half without tuned thresholds: B1 {e1:.4f} vs B2 {e2:.4f} "
               f"(tuned half: {t1:.4f} vs {t2:.4f}) -> {choice}")
    f.write_text(json.dumps({"choice": choice, "why": why}))
    say(why)
    return choice


def run(label: str, script: str, args: list, env_extra: dict) -> bool:
    env = dict(os.environ, PYTHONUTF8="1", **env_extra)
    for attempt in range(1, RETRIES + 1):
        say(f"[{label}] attempt {attempt}/{RETRIES}")
        t0 = time.time()
        with open(LOGS / f"overnight_{label}.log", "a", encoding="utf-8") as lg:
            lg.write(f"\n=== attempt {attempt} {time.strftime('%H:%M:%S')}\n")
            lg.flush()
            rc = subprocess.run([sys.executable, "-u", str(CODE / script), *args], cwd=CODE, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            say(f"[{label}] finished in {(time.time() - t0) / 60:.1f} min")
            return True
        say(f"[{label}] failed with exit code {rc} after {(time.time() - t0) / 60:.1f} min")
        time.sleep(20)
    return False


def ready(text: str):
    OUT.mkdir(exist_ok=True)
    (OUT / "READY.txt").write_text(text, encoding="utf-8")
    say("wrote READY.txt")


def main():
    dry = "--dry" in sys.argv
    keep_awake()
    LOGS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    t_start = time.time()
    say(f"autopilot started (dry={dry})")
    if dry:
        b1, b2 = held("decision_b.json"), held("decision_b2.json")
        print("B1 held-out:", {k: round(v["official"], 4) for k, v in (b1 or {}).items()})
        print("B2 held-out:", {k: round(v["official"], 4) for k, v in (b2 or {}).items()}, "| E2 state:", e2_state())
        return
    choice = choose()
    c = CONFIG[choice]
    tag = c["ce_tag"]
    pred_final = NORM / "pred_final"
    ce_env = dict(ER_CE_TAG=tag, ER_CE_MODEL_DIR=c["ce_dir"], ER_STAGE1="ranker_a.txt", ER_FEAT_TAG="_s", ER_PRED=str(NORM / "pred_a"))
    steps = [
        ("ce_test", lambda: (NORM / f"ce_test{tag}.parquet").exists(), "crossenc.py", ["score", "test"], ce_env),
        ("score", lambda: (pred_final / "_DONE").exists(), "predict.py", ["score"],
         dict(ER_CE="1", ER_CE_TAG=tag, ER_MODEL=c["ranker"], ER_PRED=str(pred_final))),
        ("write", lambda: (OUT / "candidate_pairs.tsv").exists() and (OUT / "matching_results.tsv").exists()
                          and (OUT / "candidate_pairs.tsv").stat().st_mtime > (pred_final / "_DONE").stat().st_mtime,
         "predict.py", ["write"], dict(ER_DECISION=c["decision"], ER_PRED=str(pred_final), ER_OUT=str(OUT))),
    ]
    for label, done, script, args, env in steps:
        if done():
            say(f"[{label}] already done, skipping")
            continue
        if not run(label, script, args, env):
            ready(f"FAILED at step '{label}' after {RETRIES} attempts (see logs/overnight_{label}.log).\n"
                  f"Model chosen: {choice}.\nYour validated hybrid submission is still in final_submission/matching_results.tsv.\n")
            sys.exit(1)
    # validator (organisers' script)
    vlog = OUT / "validator.log"
    if not (vlog.exists() and "PASS" in vlog.read_text(errors="replace")):
        say("[validate] running the organisers' validator")
        r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(OUT / "matching_results.tsv"),
                            "--candidate", str(OUT / "candidate_pairs.tsv"), "--test-dir", str(DATASET / "test")],
                           capture_output=True, text=True)
        vlog.write_text(r.stdout + r.stderr, encoding="utf-8")
    verdict = "PASS" if "PASS" in vlog.read_text(errors="replace") else "NOT PASSED - read validator.log"
    f = OUT / "matching_results.tsv"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    b = held(c["decision"])["early-stop half"]
    ready(
        f"FINAL SUBMISSION READY  ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
        f"File to upload : {f}\n"
        f"Size / sha256  : {f.stat().st_size / 1e6:.0f} MB / {sha}\n"
        f"Validator      : {verdict}\n\n"
        f"Model          : ranker {choice} = LightGBM with structured features + cross-encoder score ({c['ce_dir']})\n"
        f"Why            : {json.loads((MODELS / 'final_overnight.json').read_text())['why']}\n"
        f"Held-out (half whose thresholds were not tuned): recall {b['recall']:.4f}, precision {b['precision']:.4f}, "
        f"expected official score {b['official']:.4f}\n"
        f"Note           : on the previous model this simulation read about 0.011 above the leaderboard, so expect the leaderboard\n"
        f"                 to land somewhat below the held-out figure.\n\n"
        f"Earlier files  : validated hybrid (about 93% new model) final_submission/ ; first submission (0.957) output_submitted_0957/\n"
        f"Autopilot time : {(time.time() - t_start) / 3600:.1f} h this session\n")
    say("ALL DONE")


if __name__ == "__main__":
    main()
