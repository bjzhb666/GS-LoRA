export CUDA_VISIBLE_DEVICES=0
pretrain=/data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth
forget10=/data1/zhaohongbo/exps/forget-CL-pos/ratio0.190forget10lr1e-2beta0.2alpha0.005/Backbone_VIT_Epoch_63_Batch_4245_Time_2024-01-24-09-15_checkpoint.pth
forget20=/data1/zhaohongbo/exps/draw-forget-CL/CLGSLoRA/start80forgetper20lr1e-2beta0.15/task-level/Backbone_task_0.pth
forget40=/data1/zhaohongbo/exps/draw-forget-CL/CLGSLoRA/start80forgetper20lr1e-2beta0.15/task-level/Backbone_task_1.pth

python test_own.py -w 0 \
 --batch_size 2 --lora_rank 0 \
 --model ${forget40}

#  python test_own.py -w 0 \
#     --batch_size 2 --lora_rank 0 \
#     --model ${pretrain}