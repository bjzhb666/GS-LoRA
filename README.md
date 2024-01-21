# GS-LoRA for Face Transformer

All the code is in Face-Transformer folder.

## Pretrain a Face Transformer

Code is in `run_sub.sh`

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u train_own.py -b 960 -w 0,1,2,3,4,5,6,7 -d casia100 -n VIT -e 1200 \
-head CosFace --outdir ./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth18 \
--warmup-epochs 10 --lr 6e-4 --num_workers 8  --lora_rank 0 --decay-epochs 150 \
--vit_depth 18 --min-lr 2e-5
# python3 -u train.py -b 80 -w 0 -d casia \
# -n VITs -head CosFace --outdir ./results/ViT-P12S8_casia_cosface_s1 \
# --warmup-epochs 1 --lr 5e-5  --num_workers 8 -t lfw,sllfw

# python3 -u train.py -b 60 -w 0 -d casia \
# -n VITs -head CosFace --outdir ./results/ViT-P12S8_casia_cosface_s2 \
# --warmup-epochs 0 --lr 1.25e-5 -r path_to_model --num_workers 8 -t lfw,sllfw

# python3 -u train.py -b 60 -w 0 -d cas ia \
# -n VITs -head CosFace --outdir ./results/ViT-P12S8_casia_cosface_s3 \
# --warmup-epochs 0 --lr 6.25e-6 -r path_to_model  --num_workers 8 -t lfw,sllfw
```

`run.sh` is the original code using all casia dataset, not casia-100

`test.sh` is the original test code

`test_sub.sh` is the test code for our Face Transformer

## GS-LoRA

### Open vocabulary

use `run_forget_open.sh`

EWC、MAS、L2、Retrain and GS-LoRA/LoRA

See details in the bash

## Baselines

In `run_cl_forget.sh`, LIRF、SCRUB、EWC、MAS、L2、Retrain and GS-LoRA (**Main Table**)

In `run_forget.sh`, some exploration Experiment like the data ratio, group strategy, scalablity

## Exploration Experiment

Backbone forget folder gives the result of Sec 6.1
