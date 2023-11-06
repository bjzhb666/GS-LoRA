export CUDA_VISIBLE_DEVICES=2
NUM_FIRST_CLS=80
PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))
MODEL_PATH=/data1/zhaohongbo/exps/draw-forget-CL/CLGSLoRA/start80forgetper20lr1e-2beta0.15/task-level/Backbone_task_0.pth
# EWC_PATH=/data1/zhaohongbo/exps/draw-forget-CL/CL-baseline-one/EWC10-start80forget20lr1e-4/task-level/Backbone_task_0.pth
# MAS_PATH=/data1/zhaohongbo/exps/draw-forget-CL/CL-baseline-one/MAS0.01-start80forget20lr1e-4/task-level/Backbone_task_0.pth
# L2_PATH=/data1/zhaohongbo/exps/draw-forget-CL/CL-baseline-one/L20.1-start80forget20lr1e-4/task-level/Backbone_task_0.pth
EWC_PATH=/data1/zhaohongbo/exps/draw-forget-CL/onlyffn/CL-baseline-one/EWC10-start80forget20lr1e-4/task-level/Backbone_task_0.pth
MAS_PATH=/data1/zhaohongbo/exps/draw-forget-CL/onlyffn/CL-baseline-one/MAS0.01-start80forget20lr1e-4/task-level/Backbone_task_0.pth
L2_PATH=/data1/zhaohongbo/exps/draw-forget-CL/onlyffn/CL-baseline-one/L20.1-start80forget20lr1e-4/task-level/Backbone_task_0.pth

python3 -u backbone_forget_main.py -b 48 -w 0 -d casia100 -n VIT -e 50 \
    -head CosFace --outdir ./results/backbone_forget_l2_ffn \
    --warmup-epochs 0 --lr 1e-4 --num_workers 8  --lora_rank 0 --decay-epochs 150 \
    --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
    -r $L2_PATH --wandb_group backbone_forget_ours 