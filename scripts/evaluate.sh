#!/usr/bin/env bash
# Overall + per-class accuracy of a checkpoint. Replaces scripts/test_sub.sh.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CKPT=${1:?usage: $0 <checkpoint.pth> [outdir]}
OUTDIR=${2:-./exps/eval}

python -m gslora.tools.evaluate \
    -d casia100 -n VIT -head CosFace -w 0 \
    --vit_depth 6 --lora_rank 0 \
    -r "$CKPT" --outdir "$OUTDIR"
