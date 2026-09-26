"""Laptop side: pack the fine-tuned embedder (models/e5_er, about 470 MB) into e5_er.zip for upload to the SageMaker JupyterLab space.

  python pack_e5.py            -> e5_er.zip next to this file (git-ignored)

Upload it with the JupyterLab file browser's upload button (into the home folder); `bash aws/sagemaker_setup.sh` unpacks it into ~/er_work/models/.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "models" / "e5_er"
OUT = ROOT / "e5_er.zip"

if not (SRC / "model.safetensors").exists():
    raise SystemExit(f"{SRC / 'model.safetensors'} not found")
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for p in sorted(SRC.rglob("*")):
        rel = p.relative_to(SRC)
        if p.is_file() and rel.parts[0] != "ckpt":     # no training checkpoints
            z.write(p, Path("e5_er") / rel)
print(f"{OUT} ({OUT.stat().st_size / 1e6:.0f} MB): upload it to the space's home folder, then run aws/sagemaker_setup.sh")
