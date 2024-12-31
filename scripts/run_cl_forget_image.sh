#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=$(pwd):$PYTHONPATH

PRETRAIN_IMAGENET=/path/to/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth
DATA_RATIO=0.05

######################################CL baseline 4 tasks#############################################
NUM_FIRST_CLS=80
PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))
TIME=$(date "+%Y%m%d%H%M%S")
# # GS-LoRA
# for lr in 1e-2; do
#     for beta in 0.15; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir ./exps_image/CLGSLoRA/start${NUM_FIRST_CLS}forgetper${PER_FORGET_CLS}lr${lr}beta${beta}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 105 --beta $beta --alpha 0.0001 --min-lr 1e-5 --num_tasks 4 --wandb_group forget_cl_new \
#             --cl_beta_list 0.2 0.25 0.25 0.25   --prototype --pro_f_weight 0.05  --average_weight --ema_epoch 30 --ema_decay 0.9
#     done
# done
# # l2
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL/CL-baseline/L20.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --l2 --l2_lambda 0.05--wandb_group forget_clbaseline_one --replay  
#     done
# done

# # ewc
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL/CL-baseline/EWC10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --ewc --ewc_lambda 3 --replay --wandb_group forget_clbaseline  
#     done
# done

# # MAS
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL/CL-baseline/MAS0.005-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --MAS --mas_lambda 0.002 --replay --wandb_group forget_clbaseline  
#     done
# done

# # Lwf
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL/CL-baseline/Lwf10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Lwf --Lwf_lambda_remain 10 --replay --wandb_group rebuttal_forget_clbaseline   
#     done
# done

# # DER
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL/CL-baseline/DER0.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Der --DER_lambda 0.01 --replay --wandb_group newDER 
#     done
# done

# # DER++
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL/CL-baseline/DER++0.5-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Der --DER_lambda 0.01 --replay --wandb_group newDER \
#             --DER_plus --DER_plus_lambda 0.5 
#     done
# done

# # FDR
# for lr in 1e-3; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL/CL-baseline/FDR10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-8 --num_tasks 4 --one_stage --FDR --FDR_lambda 40 --replay --wandb_group rebuttal_forget_clbaseline # 
#     done
# done


# # SCRUB
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL-baseline/CL-baseline-one/SCRUB-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#             --opt adam --sgda_smoothing 0 --sgda_gamma 20 # 
#     done
# done

# # SCRUB-S
# for lr in 1e-4; do
#     for beta in 0.1; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL-baseline/CL-baseline-one/SCRUBsmooth-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#             --opt adam --sgda_smoothing 0.05 --sgda_gamma 10 # 
#     done
# done

# # retrain
# for lr in 3e-4; do
#     for beta in 0.1; do
#         for shot in 4; do
#         python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
#             -head CosFace --outdir exps_image/forget-CL-baseline/CL-baseline-one/retrain-0-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}-${TIME} \
#             --warmup-epochs 5 --lr $lr --num_workers 8 --lora_rank 0 --decay-epochs 100 \
#             --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#             -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
#             --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --retrain --l2_lambda 0 --wandb_group forget_clbaseline_one_new --replay  # \
#             # --few_shot --few_shot_num $shot
#         done
#     done
# done
