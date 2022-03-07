#!/usr/bin/env bash
datapath="<your dtu_testing path>"
outdir="<your output save path>"
resume="<your model path>"

CUDA_VISIBLE_DEVICES=0 python main.py \
        --test \
        --ndepths 48 32 8 \
        --interval_ratio 4 2 1 \
        --max_h 864 \
        --max_w 1152 \
        --num_view 5 \
        --outdir $outdir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "general_eval" \
        --batch_size 1 \
        --testlist "datasets/lists/dtu/test.txt" \
        --fea_mode "fpn" \
        --agg_mode "adaptive" \
        --depth_mode "unification" \
        --numdepth 192 \
        --interval_scale 1.06 \
        --filter_method "gipuma" \
        --prob_threshold 0.3 \
        --disp_threshold 0.25 \
        --num_consistent 3 ${@:1}
