"""Run report + deliverables of a retraining run (aws/run_full.sh, or `python main.py --retrain ...`).

  ER_ROOT=/data/er_work STATUS=SUCCESS python aws/report_full.py

Creates $ER_ROOT/deliverables/ with
  matching_results.tsv, candidate_pairs.tsv   the final files (from $ER_ROOT/output)
  RUN_REPORT.md                               status, what was trained in this run, held-out numbers per stage, timeline, sanity checks
  logs.tgz                                    every step log and the timeline
  models_manifest.tsv                         size, time and sha256 of every model this run produced
It never raises: a section that cannot be built says so in the report.
"""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

R = Path(os.environ.get("ER_ROOT", "/data/er_work"))
STATUS = os.environ.get("STATUS", "UNKNOWN")
OUT = R / "deliverables"
M, L, N = R / "models", R / "logs", R / "normalized"
CODE = Path(__file__).resolve().parent.parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"({e})"


def section(md: list, title: str, fn):
    md.append(f"\n## {title}\n")
    try:
        md.append(fn())
    except Exception as e:
        md.append(f"_could not be built: {type(e).__name__}: {e}_")


def load(name: str):
    p = M / name
    return json.loads(p.read_text()) if p.exists() else None


def held(d, key="early-stop half"):
    return None if not d else d.get("held", {}).get(key, {}).get("official")


def numbers() -> str:
    rows = ["| model | tuned half | report half (no threshold tuned on it) |", "|---|---|---|"]
    for label, f in (("ranker A (address records, no cross-encoder)", "decision_a.json"), ("ranker B (address records, + cross-encoder)", "decision_b.json")):
        d = load(f)
        rows.append(f"| {label} | {held(d, 'thr-half') or 'n/a'} | {held(d) or 'n/a'} |")
    fin = load("final.json")
    na = load("decision_na.json")
    if na:
        rows.append(f"| whole held-out set, ONE model for everyone (baseline) | {held({'held': na.get('baseline_held', {})}, 'thr-half') or 'n/a'} | {held({'held': na.get('baseline_held', {})}) or 'n/a'} |")
        rows.append(f"| **whole held-out set, address model + no-address specialist (used)** | {held(na, 'thr-half') or 'n/a'} | {held(na) or 'n/a'} |")
    out = "\n".join(rows) + f"\n\nAddress model chosen: {fin.get('choice') if fin else 'n/a'}.  Decision rule used: `{ {k: v for k, v in (na or {}).items() if k in ('thr_addr', 'thr_noaddr', 'margin', 'margin_noaddr')} }`\n"
    for name in ("embed_eval", "na_fit", "na_joint"):
        p = L / f"{name}.log"
        if p.exists():
            keep = [ln for ln in p.read_text(errors="replace").splitlines() if re.search(r"RESULT|recall|F0\.5|orphan weight|union top", ln)]
            if keep:
                out += f"\n`{name}.log`:\n```\n" + "\n".join(keep[-14:]) + "\n```\n"
    return out


def trained() -> str:
    marker = R / ".retrain_run"
    t0 = marker.stat().st_mtime if marker.exists() else None
    lines = marker.read_text().splitlines() if marker.exists() else []
    rows = [f"Components retrained in this run: `{lines[1] if len(lines) > 1 else '?'}`. Files older than the run were reused as they were.\n",
            "| file | size (MB) | written | status |", "|---|---|---|---|"]
    for p in sorted(M.rglob("*")):
        if p.is_file() and p.suffix in (".txt", ".safetensors", ".json", ".bin", ".pkl"):
            ok = ("trained in this run" if p.stat().st_mtime >= t0 else "reused (earlier run)") if t0 else "unknown (no --retrain marker)"
            rows.append(f"| {p.relative_to(M)} | {p.stat().st_size / 1e6:.1f} | {time.strftime('%m-%d %H:%M', time.localtime(p.stat().st_mtime))} | {ok} |")
    return "\n".join(rows)


def timeline() -> str:
    p = L / "timeline.tsv"
    if not p.exists():
        return "_no timeline_"
    rows, total = ["| when | step | minutes | status |", "|---|---|---|---|"], 0.0
    for ln in p.read_text().splitlines():
        a = ln.split("\t")
        if len(a) >= 4:
            rows.append(f"| {a[0]} | {a[1]} | {float(a[2]) / 60:.1f} | {a[3]} |")
            total += float(a[2])
    return "\n".join(rows) + f"\n\nTotal of the recorded steps: {total / 3600:.2f} h"


def sanity() -> str:
    import polars as pl
    m = R / "output" / "matching_results.tsv"
    if not m.exists():
        return "**matching_results.tsv is missing.**"
    d = pl.read_csv(m, separator="\t", infer_schema_length=0, quote_char=None)
    s1 = pl.read_parquet(N / "test_source1.parquet", columns=["entity_id"]).height
    n_assigned = int(d.select(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().sum()).item())
    lines = [f"- matching_results.tsv: {d.height:,} rows for {s1:,} test S1 entities ({'OK' if d.height == s1 else 'MISMATCH'}); {n_assigned:,} S2/S3 records assigned",
             f"- S1 entities with at least one match: {int((d['matched_entity_ids'].fill_null('') != '').sum()):,}"]
    v = L / "validate.log"
    lines.append("- organisers' validator (last lines):\n```\n" + "\n".join(v.read_text(errors="replace").strip().splitlines()[-6:]) + "\n```" if v.exists() else "- validator log not found")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# Retraining run: {STATUS}\n", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on `{platform.node()}` from `{R}`.  Code commit: `{sh('git -C "' + str(CODE) + '" rev-parse --short HEAD')}`.\n",
          "Pipelines: records WITH an address -> ranker A/B (address records only) + cross-encoder on the uncertain band; records WITHOUT an address -> "
          "char 3-gram TF-IDF blocking + own LightGBM, no cross-encoder. Embedder: the fine-tuned multilingual-e5-small of the earlier run, unchanged, unless listed as retrained below.\n"]
    section(md, "Models trained in this run", trained)
    section(md, "Held-out results (expected official score, per stage)", numbers)
    section(md, "Timeline", timeline)
    section(md, "Sanity checks on the final files", sanity)
    def environment() -> str:
        gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        pkgs = sh("pip list 2>/dev/null | grep -i -E '^(torch|lightgbm|polars|sentence-transformers|transformers|scikit-learn|xgboost) '")
        return "```\n" + gpu + "\npython " + platform.python_version() + "\n" + pkgs + "\n```"

    section(md, "Environment", environment)
    (OUT / "RUN_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (R / "output" / f).exists():
            shutil.copy2(R / "output" / f, OUT / f)
    with tarfile.open(OUT / "logs.tgz", "w:gz") as t:
        if L.exists():
            t.add(L, arcname="logs")
    with open(OUT / "models_manifest.tsv", "w", encoding="utf-8") as f:
        f.write("file\tbytes\tsha256\n")
        for p in sorted(M.rglob("*")):
            if p.is_file():
                f.write(f"{p.relative_to(M)}\t{p.stat().st_size}\t{sha(p)}\n")
    print(f"deliverables written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
