"""Laptop: the files Stage 1 (aws/stage1_embed.sh) needs, as ONE folder for a manual S3-console upload (Upload -> Add folder), ~1.5 GB.
  python stage1_upload.py   -> Downloads/er_stage1_upload/er_stage1   (upload to the bucket root; on SageMaker: aws s3 sync s3://<bucket>/er_stage1 ~/er_stage1)"""
import os, shutil
from pathlib import Path
REPO = Path(__file__).resolve().parent
OUT = Path.home() / "Downloads" / "er_stage1_upload" / "er_stage1"
files = [f"normalized/source{i}.parquet" for i in (1, 2, 3)] + ["normalized/eval_queries.parquet", "normalized/eval_sparse.parquet", "normalized/cepairs_rr.parquet", "normalized/ce_eval_v2.parquet"]
size = 0
for rel in files:
    src, dst = REPO / rel, OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    size += src.stat().st_size
print(f"{len(files)} files, {size / 1e9:.2f} GB -> {OUT}")
