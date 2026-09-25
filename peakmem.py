"""Run a command and report its peak resident memory (incl. children): python peakmem.py <cmd...>"""
import subprocess, sys, threading, time
import psutil

p = subprocess.Popen(sys.argv[1:])
peak = 0
def watch():
    global peak
    ps = psutil.Process(p.pid)
    while p.poll() is None:
        try:
            rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
            peak = max(peak, rss)
        except psutil.Error:
            pass
        time.sleep(0.5)
t = threading.Thread(target=watch, daemon=True); t.start()
p.wait(); t.join(1)
print(f"PEAK MEMORY: {peak / 2**30:.1f} GB  (exit {p.returncode})", flush=True)
