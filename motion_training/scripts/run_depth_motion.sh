#!/bin/bash
# moving cityscapes-Wild-256/train/aachen_1 \
# moving cityscapes-Wild-256/train/strasbourg_93
# static cityscapes-Wild-256/train/hamburg_220
# static cityscapes-Wild-256/train/aachen_2

MODEL=models/20220520_city256_softplus_resmof_smooth_sparsity_2/models/weights_39

# DS=cityscapes-Wild-256/train/aachen_1
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir $MODEL/moving_1 \
# --model_path $MODEL

# DS=cityscapes-Wild-256/train/strasbourg_93
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir $MODEL/moving_2 \
# --model_path $MODEL

# DS=cityscapes-Wild-256/train/hamburg_220
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir $MODEL/static_1 \
# --model_path $MODEL

# DS=cityscapes-Wild-256/train/aachen_2
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir $MODEL/static_2 \
# --model_path $MODEL

# DS=cityscapes-Wild-256/train/zurich_107
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir $MODEL/moving_3 \
# --model_path $MODEL

# DS=cityscapes-Wild-256/train/zurich_75
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir $MODEL/moving_4 \
# --model_path $MODEL

DS=cityscapes-Wild-256/train/bochum_55
python infer_depth_motion.py \
--input_path $DS \
--output_dir $MODEL/static_3 \
--model_path $MODEL
