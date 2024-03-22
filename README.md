# Continual Forgetting for Pre-trained Vision Models (CVPR2024)

![1711080678110](image/README/1711080678110.png)

This is the official implementation of ***GS-LoRA*** (CVPR 2024). GS-LoRA is effective, parameter-efficient, data-efficient, and easy to implement continual forgetting, where selective information is expected to be *continuously* removed from a pre-trained model while maintaining the rest. The core idea is to use LoRA combining *group* *Lasso* to realize fast model editing. For more details, please refer to:

[[2403.11530] Continual Forgetting for Pre-trained Vision Models (arxiv.org)](https://arxiv.org/abs/2403.11530)

## Getting Started

### Installation

#### a. Clone this repository

```bash
https://github.com/bjzhb666/GS-LoRA.git
cd GS-LoRA
```

#### b. Install the environment

```bash
conda create -n GSlora python=3.9
pip install -r requirements.txt
```

#### c. Prepare the datasets.

You can get our CASIA-100 in [https://drive.google.com/file/d/16CaYf45UsHPff1smxkCXaHE8tZo4wTPv/view?usp=sharing](https://drive.google.com/file/d/16CaYf45UsHPff1smxkCXaHE8tZo4wTPv/view?usp=sharing)

CASIA-100 is a subdataset from  [CASIA-WebFace](https://paperswithcode.com/dataset/casia-webface).

## Pretrain a Face Transformer

You can use our [pre-trained Face Transformer](https://drive.google.com/file/d/1kGo2eor-AYEMruyxI6_oEOegee6VB2_D/view?usp=sharing) directly.

**Or** you can train your own pre-trained models.

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

## Continual Forgetting

### Baselines

In `run_cl_forget.sh`, LIRF, SCRUB, EWC, MAS, L2, DER, DER++, FDR, LwF, Retrain and GS-LoRA (**Main Table**)

In `run_forget.sh`, some exploration Experiment like the data ratio, group strategy, scalablity

### GS-LoRA

## Exploration Experiment

#### a. Open vocabulary

use `run_forget_open.sh`

EWC、MAS、L2、Retrain and GS-LoRA/LoRA

See details in the bash

### b. Backbone forgetting

Backbone forget folder gives the result of Sec 6.1

## Citation

If you find this project useful in your research, please consider citing:

```
@article{zhao2024continual,
  title={Continual Forgetting for Pre-trained Vision Models},
  author={Zhao, Hongbo and Ni, Bolin and Wang, Haochen and Fan, Junsong and Zhu, Fei and Wang, Yuxi and Chen, Yuntao and Meng, Gaofeng and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2403.11530},
  year={2024}
}
```

## Acknowledgement

This work is built upon the [zhongyy/Face-Transformer: Face Transformer for Recognition (github.com)](https://github.com/zhongyy/Face-Transformer)

## License

This project is released under the [MIT License](https://github.com/bjzhb666/GS-LoRA/blob/master/LICENSE).
