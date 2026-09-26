"""Keeps `main.py` alive: restarts it (resumable) if it vanished without finishing and without printing a real failure."""
import os, subprocess, sys, time
import psutil

CODE = r"C:\Users\Lenovo\Downloads\Projects\ML Hackathon"
LOG = os.path.join(CODE, "run_phase2b.log")
DATASET = r"C:\Users\Lenovo\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
MAX_RESTARTS = 3


def alive() -> bool:
    for p in psutil.process_iter(["cmdline"]):
        c = p.info["cmdline"] or []
        if any(a.endswith("main.py") for a in c) and "watchdog.py" not in " ".join(c):
            return True
    return False


def say(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


restarts = 0
say("watchdog started")
while True:
    time.sleep(20)
    if alive():
        continue
    text = open(LOG, encoding="utf-8", errors="replace").read()
    tail = text[-6000:]
    if "\ntotal " in tail and "summary" in tail:
        say("main.py finished normally - watchdog done"); break
    if "\n!! " in tail:
        say("main.py stopped with a real error (see run_phase2b.log) - NOT restarting"); break
    if restarts >= MAX_RESTARTS:
        say("too many restarts - giving up"); break
    restarts += 1
    say(f"main.py vanished without finishing - restart {restarts}/{MAX_RESTARTS}")
    with open(LOG, "a", encoding="utf-8") as lg:
        subprocess.Popen([sys.executable, "-u", os.path.join(CODE, "main.py"), "--dataset", DATASET],
                         cwd=CODE, stdout=lg, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8="1"))
    time.sleep(30)
