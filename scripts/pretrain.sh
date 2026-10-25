#!/usr/bin/env bash
# Pre-train the CASIA-100 Face Transformer that the forgetting experiments
# start from. Replaces the old scripts/run_sub.sh.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WORKERS=${WORKERS:-0,1,2,3,4,5,6,7}
DEPTH=${DEPTH:-6}

python -m gslora.tools.pretrain \
    -d casia100 -n VIT -head CosFace \
    -w "$WORKERS" -b 480 -e 1200 \
    --vit_depth "$DEPTH" --lora_rank 0 \
    --lr 3e-4 --warmup-epochs 10 --decay-epochs 150 --num_workers 8 \
    --outdir "./results/ViT-P8S8_casia100_cosface_depth${DEPTH}"
