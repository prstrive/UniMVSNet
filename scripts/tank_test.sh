#!/usr/bin/env bash
datapath="<your intermediate or advanced path>"
outdir="<your output save path>"
resume="<your model path>"

CUDA_VISIBLE_DEVICES=0 python main.py \
        --test \
        --ndepths 64 32 8 \
        --interval_ratio 3 2 1 \
        --num_view 11 \
        --outdir $outdir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "general_eval" \
        --batch_size 1 \
        --testlist "all" \
        --fea_mode "fpn" \
        --agg_mode "adaptive" \
        --depth_mode "unification" \
        --numdepth 192 \
        --interval_scale 1.06 \
        --filter_method "dypcd" ${@:1}