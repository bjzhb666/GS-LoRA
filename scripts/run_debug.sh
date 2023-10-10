# GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 --master_port 26354 ./configs/r50_deformable_detr_forget.sh \
#     --batch_size 8 --output_dir exps/debug \
#     --resume /home/gaofeng_meng/zhaohongbo/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
#     --lora_rank 2 --lora_pos decoder --epoch 3 --lr 4e-4 --eval --wandb_offline

LORA_RANK=8
NUM_FIRST_CLS=75
CLS_PER_PHASE=$((80-$NUM_FIRST_CLS))
for beta in 0.008
do
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 --master_port 29500 ./configs/r50_deformable_detr_forget.sh \
                --batch_size 2 --output_dir /data2/hongbo_zhao/exps/CL-DETR-forget${CLS_PER_PHASE}-encoder-${LORA_RANK}-onestage-$beta \
                --resume /home/hongbo_zhao/Github/Deformable-DETR/exps/r50_deformable_detr-checkpoint.pth \
                --lora_rank ${LORA_RANK} --lora_pos encoder decoder --epoch 50 --lr_drop 40 --lr_gamma 0.1 --lr 2e-4 --cache_mode  \
                --lora_encoder_layers  0 1 2 3 4 5 --lora_decoder_layers  0 1 2 3 4 5 --beta $beta\
                --num_of_first_cls $NUM_FIRST_CLS --cls_per_phase $CLS_PER_PHASE --seed_cls 0 --eval --wandb_offline --no_aux_loss
done