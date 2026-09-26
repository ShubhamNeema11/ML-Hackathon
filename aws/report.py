"""Run report + deliverables folder of aws/run_pipeline.sh (also usable by hand).

  ER_ROOT=/data/er_work FINAL_DIR=/data/er_work/output_pass2 STATUS=SUCCESS python aws/report.py

Creates $ER_ROOT/deliverables/ with
  matching_results.tsv, candidate_pairs.tsv   the final files (from FINAL_DIR)
  RUN_REPORT.md                               what ran, how long, what each go/no-go gate saw, held-out numbers, sanity checks
  logs.tgz                                    every step log (real folder and mirror folder) and the timeline
  models_manifest.tsv                         size + sha256 of the models this run produced
It never raises: a section that cannot be built says so in the report.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

R = Path(os.environ.get("ER_ROOT", "/data/er_work"))
MIR = R / "fulltrain"
FINAL = Path(os.environ.get("FINAL_DIR", R / "output_pass2"))
STATUS = os.environ.get("STATUS", "UNKNOWN")
PRICE = float(os.environ.get("PRICE_PER_HOUR", 1.6))   # on-demand g5.2xlarge, ap-southeast-2 (approximate; check the current price)
OUT = R / "deliverables"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def section(title, fn):
    try:
        body = fn()
    except Exception as e:  # the report must always be written
        body = f"(could not be built: {type(e).__name__}: {e})"
    return f"\n## {title}\n\n{body}\n"


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def timeline():
    p = R / "logs" / "timeline.tsv"
    if not p.exists():
        return "no timeline recorded"
    rows = [l.split("\t") for l in p.read_text().splitlines() if l.strip()]
    tot = sum(float(r[3]) for r in rows if r[4] == "ok")
    out = ["| started | folder | step | minutes | status |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r[0]} | {'mirror' if r[1].endswith('fulltrain') else 'real'} | {r[2]} | {float(r[3]) / 60:.1f} | {r[4]} |")
    out.append(f"\nCompute time of the steps that ran: **{tot / 3600:.2f} h**. Wall-clock is shorter where CPU and GPU steps overlapped. "
               f"Estimated cost at ${PRICE:.2f}/h: **${tot / 3600 * PRICE:.0f}** (upper bound: overlapped steps are counted twice), plus storage.")
    return "\n".join(out)


def gates():
    lines = []
    g = jload(R / "logs" / "pilot_gate.json")
    if g:
        lines.append(f"- Pilot gate: **{'GO' if g['go'] else 'NO-GO'}** ({g['pairs']:,} held-out pairs; 1-AUC small {g['a_one_minus_auc']:.5f} vs pilot {g['b_one_minus_auc']:.5f}; top-1 {g['a_top1']:.4f} vs {g['b_top1']:.4f})")
    for name in ("b2", "b4"):
        d = jload(R / "models" / f"decision_{name}.json")
        if d:
            h = d["held"]
            lines.append(f"- Ranker {name.upper()} held-out expected official score: half without tuned thresholds **{h['early-stop half']['official']:.4f}**, tuned half {h['thr-half']['official']:.4f}")
    d = jload(R / "models" / "decision_pass2.json")
    if d:
        lines.append(f"- Pass 1 (same procedure): report half **{d['pass1_held']['report-half']['official']:.4f}**, tuned half {d['pass1_held']['thr-half']['official']:.4f}")
        lines.append(f"- Pass 2: report half **{d['held']['report-half']['official']:.4f}**, tuned half {d['held']['thr-half']['official']:.4f}   (decision rule: {{thr_addr {d['thr_addr']}, thr_noaddr {d['thr_noaddr']}, margin {d['margin']}}})")
    for f in ("pass2_compare.txt", "b4_compare.txt", "pilot_compare.txt", "choices.txt"):
        p = R / "logs" / f
        if p.exists():
            lines.append(f"\n`{f}`:\n```\n{p.read_text().strip()}\n```")
    lines.append("\nThe held-out simulation read about 0.008 above the leaderboard for the previous model (0.9869 vs 0.97889): expect the leaderboard to land somewhat below these numbers.")
    return "\n".join(lines) or "no gate data"


def final_files():
    m, c = FINAL / "matching_results.tsv", FINAL / "candidate_pairs.tsv"
    lines, seen, dup, empty, n, ids = [], set(), 0, 0, 0, 0
    with open(m, encoding="utf-8") as f:
        next(f)
        for line in f:
            n += 1
            _, _, rest = line.rstrip("\n").partition("\t")
            if not rest:
                empty += 1
                continue
            for i in rest.split(","):
                ids += 1
                if i in seen:
                    dup += 1
                seen.add(i)
    mean = ids / max(n, 1)
    warn = []
    if not 0.045 <= empty / n <= 0.07:
        warn.append(f"empty S1 share {empty / n:.3%} is outside the expected 4.5-7%")
    if not 3.2 <= mean <= 3.7:
        warn.append(f"mean matches per S1 {mean:.2f} is outside the expected 3.2-3.7")
    if dup:
        warn.append(f"{dup:,} record ids appear under more than one S1 (must be 0)")
    lines += [f"- `{m.name}`: {m.stat().st_size / 1e6:.0f} MB, sha256 `{sha(m)}`", f"- `{c.name}`: {c.stat().st_size / 1e9:.2f} GB, sha256 `{sha(c)}`",
              f"- S1 rows {n:,}; empty {empty:,} ({empty / n:.2%}; training singletons 5.6%); records assigned {ids:,}; mean matches per S1 {mean:.2f} (training 3.46, held-out US/India 3.4)",
              f"- sanity: {'**WARNING** ' + '; '.join(warn) if warn else 'all checks in range, every record has at most one owner'}"]
    v = FINAL / "validator.log"
    if v.exists():
        lines.append(f"- organisers' validator: `{v.read_text().strip().splitlines()[-1] if v.read_text().strip() else 'empty log'}`")
    return "\n".join(lines)


def environment():
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return "n/a"
    ver = {}
    for mod in ("polars", "lightgbm", "torch", "transformers", "sentence_transformers", "rapidfuzz"):
        try:
            ver[mod] = __import__(mod).__version__
        except Exception:
            ver[mod] = "n/a"
    gpu = run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    env = {k: v for k, v in os.environ.items() if k.startswith("ER_") or k in ("P1", "LEAN")}
    return (f"- time {time.strftime('%Y-%m-%d %H:%M:%S')}, host {platform.node()}, python {platform.python_version()}, {os.cpu_count()} vCPUs, GPU {gpu}\n"
            f"- git commit `{run('git', '-C', str(Path(__file__).resolve().parent.parent), 'rev-parse', '--short', 'HEAD')}`\n"
            f"- packages {ver}\n- settings {json.dumps(env)}")


def choices():
    ch = R / "logs" / "choices.txt"
    return ch.read_text().strip() if ch.exists() else "n/a"


def package():
    OUT.mkdir(exist_ok=True)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        src, dst = FINAL / name, OUT / name
        if src.exists() and not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    with tarfile.open(OUT / "logs.tgz", "w:gz") as t:
        for d in (R / "logs", MIR / "logs"):
            if d.exists():
                t.add(d, arcname=d.relative_to(R).as_posix())
    rows = []
    for p in sorted((R / "models").glob("*")):
        if p.is_file() and p.stat().st_mtime > time.time() - 7 * 86400:
            rows.append(f"{p.name}\t{p.stat().st_size}\t{sha(p) if p.stat().st_size < 200e6 else 'not hashed (large)'}")
    for p in sorted((R / "models").glob("ce_big")):
        if p.is_dir():
            rows.append(f"{p.name}/\t{sum(f.stat().st_size for f in p.rglob('*') if f.is_file())}\tdirectory")
    (OUT / "models_manifest.tsv").write_text("name\tbytes\tsha256\n" + "\n".join(rows) + "\n")


def main():
    head = (f"# Run report: big cross-encoder + pass-2 upgrade\n\n**Status: {STATUS}**   (final files from `{FINAL}`)\n\n"
            "Pipeline: blocking (sparse + fine-tuned e5-small dense + name-only extras) -> pass 1 (LightGBM on retrieval, fuzzy, structured and cross-encoder features) "
            "-> optional pass 2 (sibling-support re-ranking) -> tuned decision rule. No external data or services; models: multilingual-e5-small (MIT), "
            "BAAI/bge-reranker-v2-m3 (Apache-2.0), LightGBM.\n")
    body = head + section("Choices made by the automatic gates", choices) + section("Held-out results and gates", gates) \
        + section("Final files and sanity checks", final_files) + section("Step timeline", timeline) + section("Environment", environment)
    try:
        package()
    except Exception as e:
        body += f"\n(packaging failed: {e})\n"
    OUT.mkdir(exist_ok=True)
    (OUT / "RUN_REPORT.md").write_text(body, encoding="utf-8")
    print(f"report written: {OUT / 'RUN_REPORT.md'}")


if __name__ == "__main__":
    sys.exit(main())
