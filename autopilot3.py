"""Round 3, unattended: waits for the main autopilot, runs E3 (run_e3.sh), and ONLY if ranker B3 beats B2 on held-out data
scores the test set with it and writes final_overnight_v3/. Otherwise it leaves final_overnight/ (B2) as the final answer.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import autopilot as ap

OUT3 = ap.CODE / "final_overnight_v3"
GIT_BASH = "C:/Program Files/Git/bin/bash.exe"  # full path: a bare "bash" resolves to WSL, which has neither Windows Python nor our variables
MARGIN = 0.0005


def main():
    ap.keep_awake()
    OUT3.mkdir(exist_ok=True)
    ap.say("autopilot3 started: waiting for the main autopilot (final_overnight/READY.txt)")
    while not (ap.OUT / "READY.txt").exists():
        time.sleep(60)
    if "FAILED" in (ap.OUT / "READY.txt").read_text(errors="replace"):
        (OUT3 / "READY_v3.txt").write_text("Round 3 skipped: the main autopilot reported a failure (see final_overnight/READY.txt).\n")
        ap.say("main autopilot failed - round 3 skipped"); return
    ap.say("main autopilot finished: starting round 3 (E3)")
    for attempt in range(1, 4):
        with open(ap.LOGS / "overnight_e3.log", "a", encoding="utf-8") as lg:
            rc = subprocess.run([GIT_BASH, "run_e3.sh"], cwd=ap.CODE, env=dict(os.environ, ER_PYTHON=sys.executable.replace("\\", "/")),  # relative name: bash would read backslashes of a Windows path as escapes
                                     stdout=lg, stderr=subprocess.STDOUT).returncode
        ap.say(f"[e3] attempt {attempt} exit code {rc}")
        if rc == 0:
            break
    b2, b3 = ap.held("decision_b2.json"), ap.held("decision_b3.json")
    if rc != 0 or not b3:
        (OUT3 / "READY_v3.txt").write_text("Round 3 failed or produced no model (logs/overnight_e3.log). final_overnight/ (B2) remains the final answer.\n")
        ap.say("round 3 produced no model"); return
    e2, e3 = b2["early-stop half"]["official"], b3["early-stop half"]["official"]
    t2, t3 = b2["thr-half"]["official"], b3["thr-half"]["official"]
    why = f"held-out expected official, half without tuned thresholds: B2 {e2:.4f} vs B3 {e3:.4f} (tuned half {t2:.4f} vs {t3:.4f})"
    if not (e3 > e2 + MARGIN and t3 > t2 - 0.002):
        (OUT3 / "READY_v3.txt").write_text(f"Round 3 did not beat round 2 ({why}).\nfinal_overnight/ (B2) remains the final answer.\n")
        ap.say(f"B3 not better: {why}"); return
    ap.say(f"B3 is better: {why}")
    pred3 = ap.NORM / "pred_final3"
    ce_env = dict(ER_CE_TAG="_v3", ER_CE_MODEL_DIR="ce_er3", ER_STAGE1="ranker_a.txt", ER_FEAT_TAG="_s", ER_PRED=str(ap.NORM / "pred_a"))
    steps = [
        ("e3_ce_test", lambda: (ap.NORM / "ce_test_v3.parquet").exists(), "crossenc.py", ["score", "test"], ce_env),
        ("e3_score", lambda: (pred3 / "_DONE").exists(), "predict.py", ["score"],
         dict(ER_CE="1", ER_CE_TAG="_v3", ER_MODEL="ranker_b3.txt", ER_PRED=str(pred3))),
        ("e3_write", lambda: (OUT3 / "candidate_pairs.tsv").exists() and (OUT3 / "matching_results.tsv").exists()
                             and (OUT3 / "candidate_pairs.tsv").stat().st_mtime > (pred3 / "_DONE").stat().st_mtime,
         "predict.py", ["write"], dict(ER_DECISION="decision_b3.json", ER_PRED=str(pred3), ER_OUT=str(OUT3))),
    ]
    for label, done, script, args, env in steps:
        if done():
            ap.say(f"[{label}] already done"); continue
        if not ap.run(label, script, args, env):
            (OUT3 / "READY_v3.txt").write_text(f"Round 3 model was better ({why}) but step {label} failed (logs/overnight_{label}.log).\n"
                                               "final_overnight/ (B2) remains the final answer.\n")
            sys.exit(1)
    vlog = OUT3 / "validator.log"
    r = subprocess.run([sys.executable, str(ap.VALIDATOR), "--matching", str(OUT3 / "matching_results.tsv"),
                        "--candidate", str(OUT3 / "candidate_pairs.tsv"), "--test-dir", str(ap.DATASET / "test")], capture_output=True, text=True)
    vlog.write_text(r.stdout + r.stderr, encoding="utf-8")
    verdict = "PASS" if "PASS" in vlog.read_text(errors="replace") else "NOT PASSED - read validator.log"
    f = OUT3 / "matching_results.tsv"
    b = b3["early-stop half"]
    (OUT3 / "READY_v3.txt").write_text(
        f"ROUND 3 SUBMISSION READY  ({time.strftime('%Y-%m-%d %H:%M')})\n\nFile to upload : {f}\nSize / sha256  : {f.stat().st_size / 1e6:.0f} MB / "
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}\nValidator      : {verdict}\nWhy            : {why}\n"
        f"Held-out (half without tuned thresholds): recall {b['recall']:.4f}, precision {b['precision']:.4f}, expected official {b['official']:.4f}\n"
        f"Previous best  : final_overnight/matching_results.tsv (ranker B2, held-out {e2:.4f})\n")
    ap.say("ROUND 3 ALL DONE")


if __name__ == "__main__":
    main()
