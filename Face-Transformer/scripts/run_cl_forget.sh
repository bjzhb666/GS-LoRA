export CUDA_VISIBLE_DEVICES=6
NUM_FIRST_CLS=80
PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# lr=1e-2 # 1e-4?
# for lr in 1e-2
# do
# for beta in 0.15 
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
# -head CosFace --outdir /data1/zhaohongbo/exps/draw-forget-CL/CLGSLoRA/start${NUM_FIRST_CLS}forgetper${PER_FORGET_CLS}lr${lr}beta${beta} \
# --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 \
# --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
# --BND 105 --beta $beta --alpha 0.0001 --min-lr 1e-5 --num_tasks 4 --wandb_group forget_cl_new \
# --cl_beta_list 0.2 0.25 0.25 0.25 
# done
# done

# NUM_FIRST_CLS=80
# PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# lr=1e-2 # 1e-4?
# for lr in 1e-2
# do
# for beta in 0.15 
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
# -head Softmax --outdir /data1/zhaohongbo/exps/softmaxCL/forget-CL/CL/start${NUM_FIRST_CLS}forgetper${PER_FORGET_CLS}lr${lr}beta${beta} \
# --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 \
# --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_softmax_s1-1200-150de-depth6/Backbone_VIT_Epoch_508_Batch_37580_Time_2023-10-28-14-47_checkpoint.pth \
# --BND 105 --beta $beta --alpha 0.0001 --min-lr 1e-5 --num_tasks 4 --wandb_group forget_cl_new \
# --cl_beta_list 0.2 0.25 0.25 0.25
# done
# done

######################################CL baseline 1 task#############################################
# for num in 50 10
# do
# NUM_FIRST_CLS=$num
# PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# lr=1e-2 # 1e-4?
# for lr in 1e-4
# do
# for beta in 0.15 
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 1 \
# -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL/start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta} \
# --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 \
# --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
# --BND 105 --beta $beta --alpha 0.0001 --min-lr 1e-5 --num_tasks 3 --wandb_group forget_cl \
# --cl_beta_list 0.15 0.25 0.25
# done
# done

# # retrain
# for lr in 3e-4
# do
# for beta in 0.1 
# do
#     python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#         -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL-baseline/CL-baseline-one/retrain-0-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#         --warmup-epochs 5 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#         -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#         --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --retrain --l2_lambda 0 --wandb_group forget_clbaseline_one_new --replay
# done
# done

# # LIRF
# for lr in 1e-4
# do
# for beta in 0.1 
# do
#     python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#         -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL-baseline/CL-baseline-one/LIRF-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#         --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#         -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#         --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --LIRF --wandb_group forget_clbaseline_one_new --replay 
# done
# done

# # SCRUB
# for lr in 1e-4
# do
# for beta in 0.1 
# do
#     python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#         -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL-baseline/CL-baseline-one/SCRUBsmooth-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#         --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#         -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#         --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --SCRUB --wandb_group forget_clbaseline_one_new \
#         --opt adam --sgda_smoothing 0.1
# done
# done

# # l2
# for lr in 1e-4
# do
# for beta in 0.1 
# do
#     python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#         -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL-baseline/CL-baseline-one/L2-10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#         --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#         -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#         --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --l2 --l2_lambda 10 --wandb_group forget_clbaseline_one_new --replay
# done
# done

# # ewc
# for lr in 1e-4
# do
# for beta in 0.1 
# do
#     python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#         -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL-baseline/CL-baseline-one/EWC100-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#         --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#         -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#         --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --ewc --ewc_lambda 100 --replay \
#         --wandb_group forget_clbaseline_one_new 
# done
# done

# # MAS
# for lr in 1e-4
# do
# for beta in 0.1 
# do
#     python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#         -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL-baseline/CL-baseline-one/MAS1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#         --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#         -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#         --BND 110 --min-lr 1e-5 --num_tasks 1 --one_stage --MAS --mas_lambda 1 --replay \
#         --wandb_group forget_clbaseline_one_new 
# done 
# done

# done



######################################CL baseline 4 tasks#############################################
# NUM_FIRST_CLS=80
# PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# # l2
# for lr in 1e-4
# do
# for beta in 0.1 
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
# -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL-baseline/L20.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
# --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
# --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
# --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --l2 --l2_lambda 0.1 --wandb_group forget_clbaseline_one --replay
# done
# done

# # ewc
# for lr in 1e-4
# do
# for beta in 0.1 
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
# -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL-baseline/EWC10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
# --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
# --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
# --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --ewc --ewc_lambda 10 --replay --wandb_group forget_clbaseline
# done
# done

# # MAS
# for lr in 1e-4
# do
# for beta in 0.1 
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
# -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL-baseline/MAS0.005-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
# --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
# --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
# --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --MAS --mas_lambda 0.005 --replay --wandb_group forget_clbaseline
# done
# done

# # Lwf
# for lr in 1e-4
# do
# for beta in 0.1
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#     -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL-baseline/Lwf0.5-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#     --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#     --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#     -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#     --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Lwf --Lwf_lambda_remain 2 --replay --wandb_group rebuttal_forget_clbaseline
# done
# done

# # DER
# for lr in 1e-3
# do
# for beta in 0.1
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#     -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL-baseline/DER0.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#     --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#     --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#     -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#     --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Der --DER_lambda 0.01 --replay --wandb_group rebuttal_forget_clbaseline
# done
# done

# # DER++
# for lr in 1e-3
# do
# for beta in 0.1
# do
# python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#     -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL-baseline/DER0.1-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
#     --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
#     --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#     -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#     --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --Der --DER_lambda 0.001 --replay --wandb_group rebuttal_forget_clbaseline \
#     --DER_plus --DER_plus_lambda 1
# done
# done

# FDR
for lr in 1e-3
do
for beta in 0.1
do
python3 -u train_own_forget_cl.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
    -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/CL-baseline/FDR10-start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr} \
    --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 100 \
    --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
    -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
    --BND 110 --min-lr 1e-5 --num_tasks 4 --one_stage --FDR --FDR_lambda 10 --replay --wandb_group rebuttal_forget_clbaseline --wandb_offline
done
done