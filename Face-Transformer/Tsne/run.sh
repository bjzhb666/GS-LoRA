export CUDA_VISIBLE_DEVICES=0
python tsne_main.py -w 0 --batch_size 20 \
--mode before --head CosFace \
--model ./results/Tsne-depth6/Backbone_VIT_Epoch_93_Batch_7780_Time_2023-10-29-04-42_checkpoint.pth
# --lora_rank 8  --head Softmax
