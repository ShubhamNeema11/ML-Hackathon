#!/usr/bin/env bash
# Stage 1 on the laptop (RTX 4050, 6 GB): the base-size versions of the two models (multilingual-e5-base embedder, bge-reranker-base reranker),
# in its own folder C:/Users/Lenovo/er_stage1_local (links to the data; nothing of the main work folder is touched). Results: <folder>/stage1_results/.
cd "$(dirname "$0")"
export HOME=/c/Users/Lenovo ER_ROOT=/c/Users/Lenovo/er_stage1_local ER_DATASET="C:/Users/Lenovo/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset"
export ER_EMBED_BASE=intfloat/multilingual-e5-base ER_EMBED_DIR=e5b_er ER_RR_BASE=BAAI/bge-reranker-base ER_RR_BATCH=14 ER_ALLOW_SMALL_GPU=1
bash aws/stage1_embed.sh
