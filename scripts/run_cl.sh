#!/bin/sh
LORA_RANK=8
NUM_FIRST_CLS=70
CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
beta=0.013
lr=3e-4
for beta in 0.015
do
for alpha in 0.0003
do
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_forget.sh \
                --batch_size 2 --output_dir /data2/hongbo_zhao/exps-CL-debug/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE} \
                --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
                --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 30 --lr $lr --cache_mode  \
                --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
                --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
                --lora_reg_rank 0 --lora_cls_rank 0 --alpha $alpha --seed_cls 123  \
                --num_tasks 3 --cl_beta_list 0.015 0.02 0.025 
done
done

# LORA_RANK=8
# NUM_FIRST_CLS=20
# CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
# beta=0.09
# lr=3e-4
# for beta in 0.2 0.3 0.25 0.4
# do
# for alpha in 0.0003
# do
#         GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_forget.sh \
#                 --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-forget${CLS_PER_PHASE}-encoder-${LORA_RANK}-forgetcls-$beta \
#                 --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#                 --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 30 --lr $lr --cache_mode  \
#                 --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
#                 --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#                 --lora_reg_rank 0 --lora_cls_rank 0 --alpha $alpha --seed_cls 123
# done
# done

# LORA_RANK=8
# NUM_FIRST_CLS=10
# CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
# beta=0.09
# lr=3e-4
# for beta in 0.2 0.3 0.4
# do
# for alpha in 0.0003
# do
#         GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_forget.sh \
#                 --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-forget${CLS_PER_PHASE}-encoder-${LORA_RANK}-forgetcls-$beta \
#                 --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#                 --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 30 --lr $lr --cache_mode  \
#                 --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
#                 --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#                 --lora_reg_rank 0 --lora_cls_rank 0 --alpha $alpha --seed_cls 123
# done
# done


