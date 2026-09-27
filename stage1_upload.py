"""Laptop: the files Stage 1 (aws/stage1_embed.sh) needs, as ONE folder for a manual S3-console upload (Upload -> Add folder).

  python stage1_upload.py          -> Downloads/er_stage1_upload/er_stage1  (4 small files, ~55 MB; the big source files are read on
                                      SageMaker from ~/er_work of the earlier upload)
  python stage1_upload.py --full   -> the same plus source1-3 (~1.4 GB), when ~/er_work is not on the space
On SageMaker: aws s3 sync s3://<bucket>/er_stage1 ~/er_stage1 --only-show-errors
"""
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = Path.home() / "Downloads" / "er_stage1_upload" / "er_stage1"
files = ["normalized/eval_queries.parquet", "normalized/eval_sparse.parquet", "normalized/cepairs_rr.parquet", "normalized/ce_eval_v2.parquet"]
if "--full" in sys.argv:
    files += [f"normalized/source{i}.parquet" for i in (1, 2, 3)]
if OUT.exists():
    shutil.rmtree(OUT)
size = 0
for rel in files:
    src, dst = REPO / rel, OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    size += src.stat().st_size
print(f"{len(files)} files, {size / 1e6:.0f} MB -> {OUT}")
