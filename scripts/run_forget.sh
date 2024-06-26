export CUDA_VISIBLE_DEVICES=2
NUM_FIRST_CLS=90
PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# PER_FORGET_CLS=10
lr=1e-3 # 1e-4?
# for lr in 1e-2 5e-2 1e-3 
RATIO=0.1

for lr in 1e-2
do
for beta in 20
do
for alpha in 0.005
do
python3 -u train/train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
    -head CosFace  --grouping block --data_ratio $RATIO --alpha_epoch 20 \
    --outdir /path/to/exps/forget-CL-pos/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha} \
    --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 --wandb_group rebuttal_beta \
    --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
    -r /path/to/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
    --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5  --warmup_alpha --big_alpha 0.01  # --beta_decay --small_beta 1e-4
done
done
done

# # 18 layers
# for lr in 1e-2
# do
# for beta in 0.15
# do
# for alpha in 0
# do
# python3 -u train/train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#     -head CosFace --outdir /path/to/exps/forget-CL-scale/r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha} \
#     --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 --wandb_group Group_scale \
#     --vit_depth 18 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#     -r /path/to/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth18/Backbone_VIT_Epoch_1089_Batch_41360_Time_2023-11-03-14-58_checkpoint.pth \
#     --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5  --warmup_alpha --big_alpha 0.015  # --beta_decay --small_beta 1e-4
# done
# done
# done

# # LoRA pos
# NUM_FIRST_CLS=90
# PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# # PER_FORGET_CLS=10
# lr=1e-3 # 1e-4?
# # for lr in 1e-2 5e-2 1e-3 
# RATIO=0.1
# for lr in 1e-2
# do
# for beta in 0.15
# do
# for alpha in 0.005
# do
# python3 -u train/train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
#     -head CosFace  --grouping block --data_ratio $RATIO --alpha_epoch 0 \
#     --outdir /path/to/exps/forget-CL-ratio/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha} \
#     --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --lora_pos Attention --decay-epochs 100 --wandb_group lora_pos \
#     --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#     -r /path/to/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
#     --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5  --warmup_alpha --big_alpha 0.01  # --beta_decay --small_beta 1e-4
# done
# done
# done

