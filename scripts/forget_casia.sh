#!/usr/bin/env bash
# CASIA-100 forgetting: GS-LoRA, GS-LoRA++ and every baseline.
#
# Replaces run_forget.sh, run_forget_open.sh, run_cl_forget.sh and
# run_cl_forget_open.sh. The axes those four files encoded as separate scripts
# are now flags:
#     single-step vs continual   ->  --num_tasks 1  /  --num_tasks 9
#     closed vs missing-class    ->  --open_cls_num 0  /  --open_cls_num 10
#     method                     ->  --method
#
# Usage:  bash scripts/forget_casia.sh [gslora|gslora++|cl|cl-fewshot|open|<baseline>]
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PRETRAIN=${PRETRAIN:-./results/ViT-P8S8_casia100_cosface_depth6/Backbone_VIT_checkpoint.pth}
STAMP=$(date "+%Y%m%d%H%M%S")
MODE=${1:-gslora}

COMMON=(
    -d casia100 -n VIT -head CosFace
    -w 0 -b 48 --num_workers 8
    --vit_depth 6 --lora_rank 8
    -r "$PRETRAIN"
    --warmup-epochs 0 --min-lr 1e-5
)

case "$MODE" in
  # ---------------- single-step GS-LoRA ----------------
  gslora)
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method gslora -e 100 --decay-epochs 100 --lr 1e-2 \
        --num_tasks 1 --num_of_first_cls 90 --per_forget_cls 10 \
        --data_ratio 0.1 --grouping block \
        --BND 110 --beta 0.05 --alpha 0.005 \
        --warmup_alpha --big_alpha 0.01 --alpha_epoch 20 \
        --outdir "./exps/gslora-${STAMP}"
    ;;

  # ---------------- GS-LoRA++: few-shot + prototype regularisation ----------------
  gslora++)
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method gslora -e 100 --decay-epochs 100 --lr 1e-2 -b 4 \
        --num_tasks 1 --num_of_first_cls 90 --per_forget_cls 10 \
        --few_shot --few_shot_num 2 --grouping block \
        --BND 110 --beta 0.15 --alpha 0.01 \
        --warmup_alpha --big_alpha 0.01 --alpha_epoch 20 \
        --prototype --pro_f_weight 0.5 --pro_r_weight 0 --BND_pro 18 \
        --proto_aug_num 5 \
        --average_weight --ema_epoch 50 --ema_decay 0.9 \
        --outdir "./exps/gslora++-${STAMP}"
    ;;

  # ---------------- 9-task continual forgetting ----------------
  cl)
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method gslora -e 50 --decay-epochs 50 --lr 1e-2 \
        --num_tasks 9 --num_of_first_cls 90 --per_forget_cls 10 \
        --data_ratio 0.1 --grouping block \
        --BND 110 --alpha 0.0001 \
        --cl_beta_list 0.2 0.25 0.25 0.25 0.25 0.3 0.3 0.3 0.3 \
        --outdir "./exps/cl-gslora-${STAMP}"
    ;;

  cl-fewshot)
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method gslora -e 50 --decay-epochs 50 --lr 1e-2 -b 4 \
        --num_tasks 9 --num_of_first_cls 90 --per_forget_cls 10 \
        --few_shot --few_shot_num 4 --grouping block \
        --BND 110 --alpha 0.0001 \
        --cl_beta_list 0.2 0.25 0.25 0.25 0.25 0.3 0.3 0.3 0.3 \
        --prototype --pro_f_weight 0.05 --BND_pro 50 \
        --average_weight --ema_epoch 30 --ema_decay 0.9 \
        --outdir "./exps/cl-gslora++-${STAMP}"
    ;;

  # ---------------- missing-class (open) setting ----------------
  open)
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method gslora -e 50 --decay-epochs 50 --lr 1e-2 \
        --num_tasks 4 --num_of_first_cls 90 --per_forget_cls 10 \
        --open_cls_num 10 --data_ratio 0.1 --grouping block \
        --BND 110 --alpha 0.0001 --cl_beta_list 0.2 0.25 0.25 0.25 \
        --outdir "./exps/open-gslora-${STAMP}"
    ;;

  # ---------------- baselines ----------------
  l2|ewc|mas|lwf|der|fdr|scrub|lirf|retrain)
    declare -A EXTRA=(
      [l2]="--l2_lambda 0.05 --replay"
      [ewc]="--ewc_lambda 3 --replay"
      [mas]="--mas_lambda 0.002 --replay"
      [lwf]="--Lwf_lambda_remain 10 --replay"
      [der]="--DER_lambda 0.01 --replay"
      [fdr]="--FDR_lambda 40 --replay"
      [scrub]="--opt adam --sgda_smoothing 0 --sgda_gamma 20"
      [lirf]="--LIRF_T 10 --LIRF_alpha 0.1"
      [retrain]="--replay"
    )
    # shellcheck disable=SC2086
    python -m gslora.tools.train_forget "${COMMON[@]}" \
        --method "$MODE" --lora_rank 0 -e 50 --decay-epochs 50 --lr 1e-4 \
        --num_tasks 9 --num_of_first_cls 90 --per_forget_cls 10 \
        --data_ratio 0.1 --BND 110 \
        ${EXTRA[$MODE]} \
        --outdir "./exps/baseline-${MODE}-${STAMP}"
    ;;

  *)
    echo "unknown mode: $MODE" >&2
    echo "usage: $0 [gslora|gslora++|cl|cl-fewshot|open|l2|ewc|mas|lwf|der|fdr|scrub|lirf|retrain]" >&2
    exit 1
    ;;
esac
