# GS-LoRA

This repo is used to conducted the code about baselines (retrain,ewc,mas,l2) for single step forgetting and continual forgetting.

This repo can alse run continual forgetting for GS-LoRA.

## Baselines for single step forgetting or continual forgetting

### All the code is in run_cl_baseline.sh

### retrain (num_tasks=1 for single step forgetting)

```bash
NUM_FIRST_CLS=70
CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))

lr=3e-4
# retrain
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
        --batch_size 2 --output_dir /data2/hongbo_zhao/exps-CL/gpu \
        --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
        --epoch 30 --lr $lr --cache_mode  \
        --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
        --seed_cls 123  --l2_lambda 0 \
        --num_tasks 7 --retrain 
```

### EWC

```bash
# # ewc
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 39500 ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 2 --output_dir /data2/hongbo_zhao/exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-l2 \
#         --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode  \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#         --seed_cls 123  --ewc_lambda 10 \
#         --num_tasks 1 --ewc --replay 
```

### MAS

```bash
# # mas
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 2 --output_dir /data2/hongbo_zhao/exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-l2 \
#         --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode  \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#         --seed_cls 123  --mas_lambda 0.1 \
#         --num_tasks 1 --MAS  --replay 
# done
```

### L2

--replay is needed all the time, --replay denotes we will use rehearsal buffer for a fair comparison.

```bash
# # l2
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_CL_baseline.sh \
#         --batch_size 2 --output_dir /data2/hongbo_zhao/exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE}-l2 \
#         --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#         --epoch 30 --lr $lr --cache_mode  \
#         --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
#         --seed_cls 123  --l2_lambda 0.01\
#         --num_tasks 1 --l2 --replay 
```

## GS-LoRA for continual forgetting

In `run_cl.sh` 80-10-10-10-10-10-10-10 setting

```bash
LORA_RANK=8
NUM_FIRST_CLS=70
CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
beta=0.013
lr=3e-4
for beta in 0.015
do
for alpha in 0.0003
do
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 39500 ./configs/r50_deformable_detr_CL_forget.sh \
                --batch_size 2 --output_dir /data2/hongbo_zhao/exps-CL/start${NUM_FIRST_CLS}-each${CLS_PER_PHASE} \
                --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
                --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 30 --lr $lr --cache_mode  \
                --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
                --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE  --no_aux_loss \
                --lora_reg_rank 0 --lora_cls_rank 0 --alpha $alpha --seed_cls 123  \
                --num_tasks 7 --cl_beta_list 0.015 0.03 0.14 0.13 0.13 0.15 0.15
done
done
```
