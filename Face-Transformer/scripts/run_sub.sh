export CUDA_VISIBLE_DEVICES=6,4
python3 -u train_own.py -b 480 -w 0,1 -d casia100 -n VIT -e 1000 \
-head CosFace --outdir ./results/ViT-P8S8_casia100_cosface_s1-800-150de-depth6-new \
--warmup-epochs 10 --lr 3e-4 --num_workers 8  --lora_rank 0 --decay-epochs 150 \
--vit_depth 6
# python3 -u train.py -b 80 -w 0 -d casia \
# -n VITs -head CosFace --outdir ./results/ViT-P12S8_casia_cosface_s1 \
# --warmup-epochs 1 --lr 5e-5  --num_workers 8 -t lfw,sllfw

# python3 -u train.py -b 60 -w 0 -d casia \
# -n VITs -head CosFace --outdir ./results/ViT-P12S8_casia_cosface_s2 \
# --warmup-epochs 0 --lr 1.25e-5 -r path_to_model --num_workers 8 -t lfw,sllfw

# python3 -u train.py -b 60 -w 0 -d cas ia \
# -n VITs -head CosFace --outdir ./results/ViT-P12S8_casia_cosface_s3 \
# --warmup-epochs 0 --lr 6.25e-6 -r path_to_model  --num_workers 8 -t lfw,sllfw