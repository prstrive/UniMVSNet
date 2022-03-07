#!/usr/bin/env bash
datapath="<your blendedmvs path>"

resume="<your dtu checkpoint path>"
log_dir="<your log save path>"
if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2342 main.py \
        --sync_bn \
        --blendedmvs_finetune \
        --ndepths 48 32 8 \
        --interval_ratio 4 2 1 \
        --img_size 576 768 \
        --dlossw 0.5 1.0 2.0 \
        --log_dir $log_dir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "blendedmvs" \
        --nviews 7 \
        --epochs 10 \
        --batch_size 1 \
        --lr 0.0001 \
        --scheduler steplr \
        --warmup 0.2 \
        --milestones 6 8 \
        --lr_decay 0.5 \
        --trainlist "datasets/lists/blendedmvs/training_list.txt" \
        --testlist "datasets/lists/blendedmvs/validation_list.txt" \
        --fea_mode "fpn" \
        --agg_mode "adaptive" \
        --depth_mode "unification" \
        --numdepth 128 \
        --interval_scale 1.06 ${@:1} | tee -a $log_dir/log.txt
