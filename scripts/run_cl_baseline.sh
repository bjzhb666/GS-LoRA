# for num in 60 50 40 30 20 10 75 79
# do
NUM_FIRST_CLS=70
CLS_PER_PHASE=$((80 - $NUM_FIRST_CLS))

lr=6e-4
# # retrain
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 2 --output_dir ./exps-CL/gpu \
#         --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode  \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#         --seed_cls 123  --l2_lambda 0 \
#         --num_tasks 7 --retrain

# l2
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
        --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-l2 \
        --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
        --epoch 30 --lr $lr --cache_mode  \
        --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
        --seed_cls 123  --l2_lambda 0.01\
        --num_tasks 7 --l2 --replay 


# ewc
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 39500 ./configs/r50_deformable_detr_CL_baseline.sh \
        --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-ewc \
        --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
        --epoch 30 --lr $lr --cache_mode  \
        --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
        --seed_cls 123  --ewc_lambda 10 \
        --num_tasks 7 --ewc --replay

# mas
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
        --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-mas \
        --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
        --epoch 30 --lr $lr --cache_mode  \
        --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
        --seed_cls 123  --mas_lambda 0.1 \
        --num_tasks 7 --MAS  --replay

# # DER
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-DER \
#         --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode  \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#         --seed_cls 123  \
#         --num_tasks 1 --Der --DER_lambda 0.001 --wandb_offline

# # DER++
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-DERpp \
#         --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode  \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#         --seed_cls 123  \
#         --num_tasks 1 --Der --DER_lambda 0.001 --DER_plus --DER_plus_lambda 0.1

# # FDR
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-FDR \
#         --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode  \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#         --seed_cls 123  \
#         --num_tasks 7 --FDR --FDR_lambda 0.1 --wandb_offline

# # SCRUB
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh  8  --master_port 29500 \
#         ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 4 --output_dir ./exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-SCRUB \
#         --resume /data3/hongbo_zhao/CL-DETR/exps/r50_deformable_detr/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE --no_aux_loss \
#         --seed_cls 123 \
#         --num_tasks 7 --SCRUB_superepoch 3 --sgda_smoothing 0 --SCRUB  # --wandb_offline \
#          # --few_shot --few_shot_num 20
