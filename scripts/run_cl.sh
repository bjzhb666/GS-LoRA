#!/bin/sh

# GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh \
# --batch_size 8 --output_dir exps/CL-DETR

# for i in 4 8 16 32 64 
# do
#     GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr_forget.sh \
#     --batch_size 8 --output_dir exps/CL-DETR-forget40-ende-$i \
#     --resume /home/gaofeng_meng/zhaohongbo/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#     --lora_rank $i --lora_pos encoder decoder --epoch 3 --lr 4e-4
# done
# for i in 4 8 16 32 64
# do
#     GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr_forget.sh \
#         --batch_size 8 --output_dir exps/CL-DETR-forget40-encoder-$i \
#         --resume /home/gaofeng_meng/zhaohongbo/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#         --lora_rank $i --lora_pos encoder --epoch 3 --lr 4e-4
# done

# rehearsal training directly
# GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr_forget.sh \
#         --batch_size 8 --output_dir exps/CL-DETR-forget40-encoder-rehearsal_train \
#         --resume /home/gaofeng_meng/zhaohongbo/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#         --lora_rank 0 --lora_pos None --epoch 3 --lr 1e-4 --rehearsal_training --cache_mode

# declare -a lora_decoder_layers=("3 4 5" "0 1 2" "0 2 4" "1 3 5")
# # declare -a lora_decoder_layers=("0" "1" "2" "3" "4")
# # declare -a lora_encoder_layers=("3 4 5" "0 1 2" "0 2 4" "1 3 5" "4 5" "2 3" "0 1" "0 5" "5" "0" "1" "2" "3" "4")
# for layers in "${lora_decoder_layers[@]}"
# do
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_forget.sh \
#         --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-forget40-encoder-32-onestage-Lbeta \
#         --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#         --lora_rank 32 --lora_pos encoder --epoch 50 --lr 2e-4 --cache_mode \
#         --lora_encoder_layers $layers \
#         --num_of_first_cls 70 --cls_per_phase 10
# done
# GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 --master_port 39500 ./configs/r50_deformable_detr_forget.sh \
#         --batch_size 8 --output_dir /data1/gaofeng_meng/hongbo_zhao/exps/CL-DETR-forget40-encoder-8-onestage345-B0.5 \
#         --resume /home/gaofeng_meng/zhaohongbo/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#         --lora_rank 8 --lora_pos encoder --epoch 50 --lr 2e-4 --cache_mode --beta 0.5 \
#         --lora_encoder_layers 3 4 5 

# forget 30
# LORA_RANK=8
# NUM_FIRST_CLS=50
# CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
# for beta in 0.03 0.02 0.04 
# do
#         GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_forget.sh \
#                 --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-forget10-encoder-${LORA_RANK}-onestage-$beta \
#                 --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#                 --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 75  --lr_drop 50 --lr 2e-4 --cache_mode  \
#                 --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
#                 --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE 
# done


# LORA_RANK=8
# NUM_FIRST_CLS=50
# CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
# for beta in 0.06 0.07
# do
#         GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 39500 ./configs/r50_deformable_detr_forget.sh \
#                 --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-forget${CLS_PER_PHASE}-encoder-${LORA_RANK}-forgetcls-$beta \
#                 --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#                 --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 30 --lr 2e-4 --cache_mode  \
#                 --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
#                 --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE --seed_cls 0 --no_aux_loss \
#                 --lora_reg_rank 0 --lora_cls_rank 0
# done

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
                --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-forget${CLS_PER_PHASE}-en+de-${LORA_RANK}-forgetcls-$beta \
                --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
                --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 30 --lr $lr --cache_mode  \
                --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
                --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
                --lora_reg_rank 0 --lora_cls_rank 0 --alpha $alpha --seed_cls 123 \
                --num_tasks 2
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


