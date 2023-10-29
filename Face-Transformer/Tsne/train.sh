export CUDA_VISIBLE_DEVICES=0
python3 -u train_own.py -b 96 -w 0 -d tsne -n VIT -e 500 \
-head CosFace --outdir ./results/Tsne-depth6-500ep \
--warmup-epochs 5 --lr 3e-4 --num_workers 8  --lora_rank 0 --decay-epochs 400 \
--vit_depth 6