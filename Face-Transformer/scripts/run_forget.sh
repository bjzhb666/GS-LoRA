export CUDA_VISIBLE_DEVICES=2
NUM_FIRST_CLS=90
PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# PER_FORGET_CLS=10
lr=1e-3 # 1e-4?
# for lr in 1e-2 5e-2 1e-3 
for lr in 1e-2
do
for beta in 0.15
do
for alpha in 0.0001
do
python3 -u train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
-head CosFace --outdir /data1/zhaohongbo/exps/forget-CL/start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha} \
--warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 --wandb_group Group_sparse \
--vit_depth 12 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
-r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth12-new/Backbone_VIT_Epoch_1119_Batch_82760_Time_2023-10-28-05-35_checkpoint.pth \
--BND 110 --beta $beta --alpha $alpha --min-lr 1e-5  # --warmup_alpha --big_alpha 0.1  --beta_decay --small_beta 1e-4
done
done
done


# for lr in 1e-2
# do
# for beta in 0.25
# do
# for alpha in 0.0001
# do
# python3 -u train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e 100 \
# -head Softmax --outdir /data1/zhaohongbo/exps/softmaxsingle/forget-CL/start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta} \
# --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 --wandb_group softmax-single \
# --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_softmax_s1-1200-150de-depth6/Backbone_VIT_Epoch_508_Batch_37580_Time_2023-10-28-14-47_checkpoint.pth \
# --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5  # --warmup_alpha --big_alpha 0.1  --beta_decay --small_beta 1e-4
# done
# done
# done
