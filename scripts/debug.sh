#!/usr/bin/env bash
NUM_FIRST_CLS=70
CLS_PER_PHASE=$((80 - $NUM_FIRST_CLS))
lr=3e-4
LORA_RANK=8

./configs/r50_deformable_detr_CL_forget.sh \
        --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-DEBUG \
        --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
        --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 30 --lr $lr --cache_mode \
        --lora_encoder_layers 0 1 2 3 4 5 --lora_decoder_layers 0 1 2 3 4 5 --beta 0.015 --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE --no_aux_loss \
        --lora_reg_rank 0 --lora_cls_rank 0 --alpha 0.0003 --seed_cls 123 \
        --num_tasks 1 --cl_beta_list 0.015 0.03 0.14 0.13 0.13 0.15 0.15 \
                --wandb_offline  --few_shot --few_shot_num 20