"""Laptop: the files the AWS part of Stage 2 (aws/stage2_gpu.sh) needs beyond Stage 1's, as ONE folder for the S3 console (~1.4 GB).
  python stage2_upload.py   -> Downloads/er_stage2_upload/er_stage2   (upload to the bucket root; on SageMaker:
                               aws s3 sync s3://<bucket>/er_stage2 ~/er_stage1 --only-show-errors   - into the Stage-1 folder)"""
import os, shutil
from pathlib import Path
REPO = Path(__file__).resolve().parent
OUT = Path.home() / "Downloads" / "er_stage2_upload" / "er_stage2"
files = [f"normalized/test_source{i}.parquet" for i in (1, 2, 3)] + ["normalized/train_queries.parquet", "normalized/trainall_queries.parquet",
                                                                     "normalized/ce_train_v2.parquet", "normalized/ce_test_v2.parquet"]
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
print(f"{len(files)} files, {size / 1e9:.2f} GB -> {OUT}")
