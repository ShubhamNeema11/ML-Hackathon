"""Build fulltrain/: a mirror work folder in which the TRAINING records play the role of the test set.

Running the unchanged test pipeline there (block -> dense -> extras -> ranker A -> cross-encoder -> ranker B2) gives the
full-density candidate table with pass-1 probabilities for every training record. Because the held-out S1 entities
(is_val_s1) were never used to train any model, their rows are out-of-sample, and the true per-S1 macro F0.5 can be computed
exactly on them (no simulation). Nothing is copied: files are hard links.

    ER_ROOT=/data/er_work python fulltrain_setup.py
"""
import os
from pathlib import Path

SRC = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parent))   # the real work folder (normalized/, models/)
DST = SRC / "fulltrain"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def main():
    for i in (1, 2, 3):
        link(SRC / "normalized" / f"source{i}.parquet", DST / "normalized" / f"test_source{i}.parquet")
        fix = SRC / "normalized" / f"state_fix_source{i}.parquet"   # the corrected-state overlay of the training files plays the test overlay here
        if fix.exists():
            link(fix, DST / "normalized" / f"state_fix_test_source{i}.parquet")
    for f in (SRC / "models").rglob("*"):
        if f.is_file():
            link(f, DST / "models" / f.relative_to(SRC / "models"))
    (DST / "logs").mkdir(exist_ok=True)
    print("fulltrain/ ready:", sorted(p.name for p in (DST / "normalized").iterdir()))


if __name__ == "__main__":
    main()
