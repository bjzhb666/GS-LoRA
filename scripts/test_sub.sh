export CUDA_VISIBLE_DEVICES=0
pretrain=./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth

python test/test_own.py -w 0 \
 --batch_size 2 --lora_rank 0 \
 --model ${pretrain}
