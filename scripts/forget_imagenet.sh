#!/usr/bin/env bash
# ImageNet-100 continual forgetting on torchvision ViT-B/16.
# Replaces run_cl_forget_image.sh.
#
# Usage: bash scripts/forget_imagenet.sh [gslora|l2|ewc|mas|lwf|der|fdr|scrub|retrain]
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# The raw torchvision checkpoint; torch.hub puts it here after the first
# vit_b_16(weights=IMAGENET1K_V1) call.
PRETRAIN=${PRETRAIN:-$HOME/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth}
STAMP=$(date "+%Y%m%d%H%M%S")
MODE=${1:-gslora}

COMMON=(
    -d imagenet100 -n VIT_B16 -head CosFace
    -w 0 -b 48 --num_workers 8
    -r "$PRETRAIN"
    --num_tasks 4 --num_of_first_cls 80 --per_forget_cls 20
    --data_ratio 0.05 --warmup-epochs 0 --min-lr 1e-5
    -e 100 --decay-epochs 100
)

case "$MODE" in
  gslora)
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method gslora --lora_rank 8 --lr 1e-2 \
        --BND 105 --alpha 0.0001 --cl_beta_list 0.2 0.25 0.25 0.25 \
        --prototype --pro_f_weight 0.05 \
        --average_weight --ema_epoch 30 --ema_decay 0.9 \
        --outdir "./exps_image/gslora-${STAMP}"
    ;;
  l2|ewc|mas|lwf|der|fdr|scrub|retrain)
    declare -A EXTRA=(
      [l2]="--l2_lambda 0.05 --replay"
      [ewc]="--ewc_lambda 3 --replay"
      [mas]="--mas_lambda 0.002 --replay"
      [lwf]="--Lwf_lambda_remain 10 --replay"
      [der]="--DER_lambda 0.01 --replay"
      [fdr]="--FDR_lambda 40 --replay"
      [scrub]="--opt adam --sgda_smoothing 0 --sgda_gamma 20"
      [retrain]="--replay"
    )
    # shellcheck disable=SC2086
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method "$MODE" --lora_rank 0 --lr 1e-4 --BND 110 \
        ${EXTRA[$MODE]} \
        --outdir "./exps_image/baseline-${MODE}-${STAMP}"
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
