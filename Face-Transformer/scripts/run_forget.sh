export CUDA_VISIBLE_DEVICES=6,4
NUM_FIRST_CLS=90
PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
lr=3e-4 # 1e-4?
for beta in 0.001 0.0001 0.005 
do
python3 -u train_own_forget.py -b 480 -w 0,1 -d casia100 -n VIT -e 400 \
-head CosFace --outdir ./results/forget/start${NUM_FIRST_CLS}forget${PER_FORGET_CLS} \
--warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 \
--vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
-r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth \
--BND 1e10 --beta $beta --alpha 0.0003
done