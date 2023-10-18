export CUDA_VISIBLE_DEVICES=6,5,4
NUM_FIRST_CLS=90
PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
python3 -u train_own_forget.py -b 60 -w 0,1,2 -d casia100 -n VIT -e 1 \
-head CosFace --outdir ./results/forget/start${NUM_FIRST_CLS}forget${PER_FORGET_CLS} \
--warmup-epochs 0 --lr 3e-4 --num_workers 8  --lora_rank 8 --decay-epochs 150 \
--vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
--wandb_offline -r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-800-150de-depth6-new/Backbone_VIT_Epoch_989_Batch_73180_Time_2023-10-18-01-21_checkpoint.pth \
--BND 20 --beta 0.03 --alpha 0.0003