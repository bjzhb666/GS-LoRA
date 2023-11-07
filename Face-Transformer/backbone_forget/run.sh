export CUDA_VISIBLE_DEVICES=1
NUM_FIRST_CLS=90
PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
# MODEL_PATH=/data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/backbone_forget/masking.pth
# MODEL_PATH=/data1/zhaohongbo/exps/forget-CL/r128start90forget10lr1e-2beta0.15alpha0/Backbone_VIT_Epoch_54_Batch_3620_Time_2023-11-01-01-29_checkpoint.pth 
# MODEL_PATH=/data1/zhaohongbo/exps/forget-CL/r8start90forget10lr1e-2beta0.15alpha0/Backbone_VIT_Epoch_55_Batch_3730_Time_2023-11-02-00-07_checkpoint.pth
# MODEL_PATH=/data1/zhaohongbo/exps/forget-CL/r64start90forget10lr1e-2beta0.15alpha0/Backbone_VIT_Epoch_27_Batch_1810_Time_2023-11-01-12-00_checkpoint.pth

# 大稀疏的rank8
# MODEL_PATH=/data1/zhaohongbo/exps/forget-CL-backbone/ratio0.1rank8start90forget10lr1e-2beta0.15alpha0/Backbone_VIT_Epoch_44_Batch_2980_Time_2023-11-06-08-49_checkpoint.pth
# 大稀疏的rank64
MODEL_PATH=/data1/zhaohongbo/exps/forget-CL-backbone/ratio0.1rank64start90forget10lr1e-2beta0.15alpha0/Backbone_VIT_Epoch_27_Batch_1810_Time_2023-11-06-07-27_checkpoint.pth
lr=1e-4
# lr=1e-2
python3 -u backbone_forget_main.py -b 48 -w 0 -d casia100 -n VIT -e 50 \
    -head CosFace --outdir /data1/zhaohongbo/exps/forget-CL-backbone/resultsr64sparser-50ep-lr${lr} \
    --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 0 --decay-epochs 150 \
    --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
    -r $MODEL_PATH --wandb_group backbone_forget_ours # --opt sgd