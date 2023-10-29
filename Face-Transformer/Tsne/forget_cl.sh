export CUDA_VISIBLE_DEVICES=0
NUM_FIRST_CLS=8
PER_FORGET_CLS=$((10-$NUM_FIRST_CLS))
lr=1e-2 # 1e-4?
for lr in 1e-2
do
for beta in 0.15 
do
python3 -u train_own_forget_cl.py -b 48 -w 0 -d tsne -n VIT -e 200 \
-head CosFace --outdir ./results/Tsne-depth6/200epcl025025 \
--warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 150 \
--vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
-r /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/Tsne-depth6/Backbone_VIT_Epoch_93_Batch_7780_Time_2023-10-29-04-42_checkpoint.pth \
--BND 105 --beta $beta --alpha 0.0001 --min-lr 1e-5 --num_tasks 2 --wandb_group forget_cl_new \
--cl_beta_list 0.25 0.25
done
done