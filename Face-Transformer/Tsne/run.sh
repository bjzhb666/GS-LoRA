export CUDA_VISIBLE_DEVICES=5
python tsne_main.py -w 0 --batch_size 20 \
--mode before --head Softmax \
--model /data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_softmax_s1-1200-150de-depth6/Backbone_VIT_Epoch_508_Batch_37580_Time_2023-10-28-14-47_checkpoint.pth
# --lora_rank 8  --head Softmax
