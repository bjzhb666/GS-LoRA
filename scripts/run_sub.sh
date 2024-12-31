export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -u train/train_own.py -b 480 -w 0,1,2,3,4,5,6,7 -d casia100 -n VIT -e 1200 \
    -head CosFace --outdir ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth12new-bs480 \
    --warmup-epochs 10 --lr 3e-4 --num_workers 8  --lora_rank 0 --decay-epochs 150 \
    --vit_depth 12  

# python3 -u train/train_own.py -b 1920 -w 0,1,2,3,4,5,6,7 -d casia100 -n VIT -e 1200 \
#     -head Softmax --outdir ./results/ViT-P8S8_casia100_softmax_s1-1200-150de-depth12new \
#     --warmup-epochs 10 --lr 12e-4 --num_workers 8  --lora_rank 0 --decay-epochs 150 \
#     --vit_depth 12 --min-lr 2e-5

# python3 -u train/train_own.py -b 960 -w 0,1,2,3,4,5,6,7 -d casia100 -n VIT -e 1200 \
#     -head CosFace --outdir ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth18new \
#     --warmup-epochs 10 --lr 6e-4 --num_workers 8  --lora_rank 0 --decay-epochs 150 \
#     --vit_depth 18 --min-lr 2e-5 