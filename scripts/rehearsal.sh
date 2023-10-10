#!/bin/sh

# rehearsal training
LORA_RANK=8
NUM_FIRST_CLS=79
beta=0.013
lr=2e-4
for NUM_FIRST_CLS in 20 10
do
CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_forget.sh \
                --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-rehearal${CLS_PER_PHASE} \
                --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
                --lora_rank 0 --lora_pos None  --lr $lr --cache_mode  \
                --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE   \
                --rehearsal_training --finetune_rehearsal_epoch 50 --one_stage \
                --seed_cls 123
done
# --one_stage action='store_false'

# finetune gradient ascent
LORA_RANK=8
NUM_FIRST_CLS=79
beta=0.013
lr=2e-4
for NUM_FIRST_CLS in 79 75 70 60 50 40 30 20 10
do
CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_forget.sh \
                --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-gradient-ascent${CLS_PER_PHASE} \
                --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
                --lora_rank 0 --lora_pos None --lr $lr --cache_mode  \
                --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE   \
                --finetune_gradient_ascent --finetune_forget_epoch 5 --one_stage \
                --seed_cls 123
done
# --one_stage action='store_false'