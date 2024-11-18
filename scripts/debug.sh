#!/usr/bin/env bash
NUM_FIRST_CLS=70
CLS_PER_PHASE=$((80 - $NUM_FIRST_CLS))
lr=3e-4

./configs/r50_deformable_detr_CL_baseline.sh \
        --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-SCRUB \
        --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
        --epoch 30 --lr $lr --cache_mode \
        --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE --no_aux_loss \
        --seed_cls 123 \
        --num_tasks 1 --SCRUB_superepoch 3 --sgda_smoothing 0 --SCRUB --wandb_offline \
        --few_shot --few_shot_num 20