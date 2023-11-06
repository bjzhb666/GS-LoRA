export CUDA_VISIBLE_DEVICES=0
After_PATH=/data1/zhaohongbo/exps/draw-forget-CL/CLGSLoRA/start80forgetper20lr1e-2beta0.15/task-level/Backbone_task_0.pth
Before_PATH=/data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth

python tsne_main.py -w 0 --batch_size 20 \
--mode before --head CosFace \
--model $After_PATH
# --lora_rank 8  --head Softmax
# --model ./results/Tsne-depth6/Backbone_VIT_Epoch_93_Batch_7780_Time_2023-10-29-04-42_checkpoint.pth