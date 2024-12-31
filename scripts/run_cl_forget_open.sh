#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$(pwd):$PYTHONPATH
# NUM_FIRST_CLS=80
# PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))
# TIME=$(date "+%Y%m%d%H%M%S")

######################################CL baseline 1 task#############################################

# # retrain
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 3e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/retrain-0-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 5 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --retrain --l2_lambda 0 --wandb_group forget_clbaseline_one_new --replay --few_shot --few_shot_num $shot
#         done
#     done
# done

# # LIRF
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/LIRF-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --LIRF --wandb_group forget_clbaseline_one_new --replay  --few_shot --few_shot_num $shot
#         done
#     done
# done

# # Lwf10
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/Lwf0.05-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --Lwf --Lwf_lambda_remain 2 --replay --wandb_group rebuttal_forget_clbaseline  --few_shot --few_shot_num 4
#     done
# done

# # Lwf50
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/Lwf0.05-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --Lwf --Lwf_lambda_remain 10 --replay --wandb_group rebuttal_forget_clbaseline  --few_shot --few_shot_num 4
#     done
# done

# # SCRUB
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/SCRUBsmooth-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#                 --opt adam --sgda_smoothing 0 --few_shot --few_shot_num $shot
#         done
#     done
# done

# # SCRUB50
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/SCRUBsmooth-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#                 --opt adam --sgda_smoothing 0 --sgda_gamma 2 --few_shot --few_shot_num $shot
#         done
#     done
# done

# # SCRUB-S for some case --sgda_gamma 3
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/SCRUBsmooth-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#                 --opt adam --sgda_smoothing 0.05   --few_shot --few_shot_num $shot
#         done
#     done
# done

# # l2
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/L2-0.05-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --l2 --l2_lambda 0.05 --wandb_group forget_clbaseline_one_new --replay --few_shot --few_shot_num $shot
#         done
#     done
# done

# # ewc
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/EWC10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --ewc --ewc_lambda 10 --replay \
#                 --wandb_group forget_clbaseline_one_new --few_shot --few_shot_num $shot
#         done
#     done
# done

# # MAS
# for lr in 1e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/MAS5-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --MAS --mas_lambda 5 --replay \
#                 --wandb_group forget_clbaseline_one_new --few_shot --few_shot_num $shot
#         done
#     done
# done

# # FDR
# for lr in 1e-3; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL/CL-baseline/FDR10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --FDR --FDR_lambda 50 --replay --wandb_group rebuttal_forget_clbaseline --few_shot --few_shot_num $shot
#         done
#     done
# done

# # DER
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-3; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL/CL-baseline/DER0.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --Der --DER_lambda 0.01 --replay --wandb_group rebuttal_forget_clbaseline  --few_shot --few_shot_num $shot
#         done
#     done
# done

# # DER++
# for lr in 1e-3; do
#     for beta in 0.1; do
#         for shot in 4; do
#             python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#                 -head CosFace --outdir exps/forget-CL/CL-baseline/DER++0.5-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#                 --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#                 --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#                 -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#                 --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --Der --DER_lambda 0.05 --replay --wandb_group rebuttal_forget_clbaseline \
#                 --DER_plus --DER_plus_lambda 0.005  --few_shot --few_shot_num $shot
#         done
#     done
# done

######################################CL baseline 4 tasks#############################################
NUM_FIRST_CLS=80
PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))
TIME=$(date "+%Y%m%d%H%M%S")
# GS-LoRA
for lr in 1e-2; do
    for beta in 0.15; do
        python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
            -head CosFace --outdir ./exps/CLGSLoRAopen/start${NUM_FIRST_CLS}forgetper${PER_FORGET_CLS}lr${lr}beta${beta}-${TIME} \
            --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs 100 \
            --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
            -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
            --BND 105 --beta $beta --alpha 0.0001 --min-lr 1e-5 --num_tasks 4 --wandb_group forget_cl_open --warmup_alpha --big_alpha 0.015 --alpha_epoch 30 \
            --cl_beta_list 0.14 0.14 0.2 0.15 --prototype --pro_f_weight 0.0003
    done
done

# # l2
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/L20.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --l2 --l2_lambda 0.1 --wandb_group forget_clbaseline_one --replay  # --few_shot --few_shot_num 4
#     done
# done

# # ewc
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/EWC10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --ewc --ewc_lambda 10 --replay --wandb_group forget_clbaseline  # --few_shot --few_shot_num 4
#     done
# done

# # MAS
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/MAS0.005-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --MAS --mas_lambda 0.005 --replay --wandb_group forget_clbaseline  # --few_shot --few_shot_num 4
#     done
# done

# # Lwf
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/Lwf0.05-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Lwf --Lwf_lambda_remain 10 --replay --wandb_group rebuttal_forget_clbaseline   # --few_shot --few_shot_num 4
#     done
# done

# # DER
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-3; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/DER0.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Der --DER_lambda 0.01 --replay --wandb_group rebuttal_forget_clbaseline # --few_shot --few_shot_num 4
#     done
# done

# # DER++
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-3; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/DER++0.5-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Der --DER_lambda 0.005 --replay --wandb_group rebuttal_forget_clbaseline \
#             --DER_plus --DER_plus_lambda 0.005 # --few_shot --few_shot_num 4
#     done
# done

# # FDR
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-3; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL/CL-baseline/FDR10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --FDR --FDR_lambda 50 --replay --wandb_group rebuttal_forget_clbaseline # --few_shot --few_shot_num 4
#     done
# done

# # LIRF
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/LIRF-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --LIRF --wandb_group forget_clbaseline_one_new --replay # --few_shot --few_shot_num 4
#     done
# done

# # SCRUB
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/SCRUB-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#             --opt adam --sgda_smoothing 0 # --few_shot --few_shot_num 4
#     done
# done

# # SCRUB-S
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/SCRUBsmooth-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#             --opt adam --sgda_smoothing 0.1 # --few_shot --few_shot_num 4
#     done
# done

# # retrain
# TIME=$(date "+%Y%m%d%H%M%S")
# for lr in 3e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#         python3 -u train/train_own_forget_cl_open.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#             -head CosFace --outdir exps/forget-CL-baseline/CL-baseline-one/retrain-0-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 5 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --retrain --l2_lambda 0 --wandb_group forget_clbaseline_one_new --replay  # \
#             # --few_shot --few_shot_num $shot
#         done
#     done
# done
