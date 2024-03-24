export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u train/train_own.py -b 960 -w 0,1,2,3,4,5,6,7 -d casia100 -n VIT -e 1200 \
    -head CosFace --outdir ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6 \
    --warmup-epochs 10 --lr 6e-4 --num_workers 8  --lora_rank 0 --decay-epochs 150 \
    --vit_depth 6 --min-lr 2e-5
