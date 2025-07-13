#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$(pwd):$PYTHONPATH
NUM_FIRST_CLS=90
PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))
# PER_FORGET_CLS=10
# lr=1e-3 # 1e-4?
# for lr in 1e-2 5e-2 1e-3
EPOCH=100
RATIO=0.1
RANK=8

TIME=$(date "+%Y%m%d%H%M%S")

# for lr in 1e-2; do
#     for shot in 4; do
#         for beta in 0.05; do
#             for alpha in 0.005; do
#                 for weight in 0.005; do
#                     python3 -u train/train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e $EPOCH \
#                         -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 20 \
#                         --outdir ./exps/forget-CL-pos/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}epoch${EPOCH}-${TIME} \
#                         --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank $RANK --decay-epochs $EPOCH --wandb_group data \
#                         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                         -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                         --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5 --warmup_alpha --big_alpha 0.01  \
#                           --prototype --pro_f_weight $weight --pro_r_weight 0  --average_weight --ema_epoch 50 --ema_decay 0.9
#                 done
#             done
#         done
#     done
# done

# few shot
for lr in 1e-2; do
    for shot in 2; do
        for beta in 0.15; do
            for alpha in 0.01; do
                for fpweight in 0.5; do
                    for rpweight in 0; do
                        python3 -u train/train_own_forget.py -b 4 -w 0 -d casia100 -n VIT -e $EPOCH \
                            -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 20 \
                            --outdir ./exps/forget-CL-pos/ratio${RATIO}r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}epoch${EPOCH}-${TIME}-fpweight${fpweight}-shot${shot} \
                            --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs $EPOCH --wandb_group fewshot \
                            --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
                            -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
                            --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5 --warmup_alpha --big_alpha $alpha \
                            --prototype --pro_f_weight $fpweight --pro_r_weight 0 --average_weight --ema_epoch 50 --ema_decay 0.9 \
                            --few_shot --few_shot_num $shot --wandb_name Aug5prototype-Few${shot}start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}epoch${EPOCH}-fpweight${fpweight}-rpweight${rpweight} \
                            --aug_num 5
                    done
                done
            done
        done
    done
done

# # 12 layers
# for lr in 1e-2; do
#     for shot in 4; do
#         for beta in 0.03; do
#             for alpha in 0.005; do
#                 for weight in 0.001; do
#                     python3 -u train/train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e $EPOCH \
#                         -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 20 \
#                         --outdir ./exps/forget-CL-pos/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}epoch${EPOCH}-${TIME} \
#                         --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs $EPOCH --wandb_group rebuttal_beta \
#                         --vit_depth 12 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                         -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth12new-bs480/Backbone_VIT_Epoch_1111_Batch_83320_Time_2024-12-13-17-21_checkpoint.pth \
#                         --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5 --warmup_alpha --big_alpha 0.01  \
#                           --prototype --pro_f_weight $weight --pro_r_weight 0.001   --average_weight --ema_epoch 50 --ema_decay 0.9
#                     # --beta_decay --small_beta 1e-4 --few_shot --few_shot_num $shot
#                 done
#             done
#         done
#     done
# done
