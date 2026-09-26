"""Laptop side, no Git Bash needed: build ONE folder with exactly the files run_quick.sh needs, for a manual upload in the S3 console.

  python stage_upload.py        -> C:/Users/Lenovo/Downloads/er_work_upload/er_work   (485 files, about 7.75 GB; hard links, no extra disk)

In the S3 console: open the bucket (name starting with "sagemaker-"), Upload -> Add folder -> choose that er_work folder -> Upload.
The bucket then holds er_work/models/... and er_work/normalized/..., which run_quick.sh syncs to ~/er_work on SageMaker.
The file list is the one of upload_quick.sh (the same files, uploaded by the AWS CLI instead).
"""
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = Path(os.environ.get("ER_STAGE", Path.home() / "Downloads" / "er_work_upload" / "er_work"))


def main():
    s = (REPO / "upload_quick.sh").read_text(encoding="utf-8")
    files = re.search(r"FILES=\((.*?)\)", s, re.S).group(1).split()
    dirs = re.search(r"DIRS=\((.*?)\)", s, re.S).group(1).split()
    n = size = 0
    for rel in files + [str(p.relative_to(REPO)) for d in dirs for p in (REPO / d).rglob("*") if p.is_file() and "ckpt" not in p.parts]:
        src, dst = REPO / rel, OUT / rel
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:          # another drive: copy instead
                shutil.copy2(src, dst)
        n += 1
        size += src.stat().st_size
    print(f"{n} files, {size / 1e9:.2f} GB -> {OUT}\nupload that er_work folder to the bucket root with the S3 console (Upload -> Add folder)")


if __name__ == "__main__":
    main()
