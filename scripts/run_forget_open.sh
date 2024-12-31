#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=$(pwd):$PYTHONPATH

NUM_FIRST_CLS=70
PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))
# PER_FORGET_CLS=10
lr=1e-3 # 1e-4?
# for lr in 1e-2 5e-2 1e-3
RATIO=0.1
OPEN=40

# # GS-LoRA/LoRA (depends on alpha=0 or not)
# for lr in 1e-2; do
#     for beta in 0.1; do
#         for alpha in 0; do
#             python3 -u train_own_forget_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 0 --open_cls_num $OPEN \
#                 --outdir ./exps/forget-CL-ratio/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}open${OPEN} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs 100 --wandb_group forget-open \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#                 --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5 # --warmup_alpha --big_alpha 0.009 # --beta_decay --small_beta 1e-4
#         done
#     done
# done

# L2
for lr in 1e-4; do
    for beta in 0.1; do
        for alpha in 0; do
            python3 -u train/train_own_forget_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
                -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 0 --open_cls_num $OPEN \
                --outdir ./exps/forget-CL-open/L2-0.01-ratio${RATIO}$start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}open${OPEN} \
                --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 --wandb_group forget-open-baseline-new \
                --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
                -r ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
                --min-lr 1e-5 --one_stage --l2 --l2_lambda 0.01 --replay --wandb_offline
        done
    done
done

# # EWC
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for alpha in 0; do
#             python3 -u train_own_forget_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 0 --open_cls_num $OPEN \
#                 --outdir ./exps/forget-CL-open/EWC10-ratio${RATIO}$start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}open${OPEN} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 --wandb_group forget-open-baseline-new \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#                 --min-lr 1e-5 --one_stage --ewc --ewc_lambda 10 --replay
#         done
#     done
# done

# # MAS

# for lr in 1e-4; do
#     python3 -u train_own_forget_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#         -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 0 --open_cls_num $OPEN \
#         --outdir ./exps/forget-CL-open/MAS1-ratio${RATIO}$start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}open${OPEN} \
#         --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 --wandb_group forget-open-baseline-new \
#         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#         -r ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#         --min-lr 1e-5 --one_stage --MAS --mas_lambda 1 --replay
# done

# # retrain
# for lr in 3e-4; do
#     for beta in 0.1; do
#         for alpha in 0; do
#             python3 -u train_own_forget_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 0 --open_cls_num $OPEN \
#                 --outdir ./exps/forget-CL-open/retrain-ratio${RATIO}$start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}open${OPEN} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 --wandb_group forget-open-baseline-new \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#                 --min-lr 1e-5 --one_stage --retrain --replay
#         done
#     done
# done
