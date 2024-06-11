#!/bin/bash
python train.py \
--only_ego_motion \
--gpus_to_use 2 \
--data_path /seokju4/Dataset/fei-Dataset/cityscapes-Wild-128/train/ \
--png \
--weighted_ssim \
--boxify \
--model_name ./debug_city3/ \
--seg_mask none \
--log_frequency 500 \
--save_frequency 2 \
--batch_size 12 \
--log_depth \
--log_lr \
--log_multiframe \
--log_trans whole \
--reconstr_loss_weight 0.8 \
--ssim_loss_weight 1 \
--smooth_loss_weight 1e-3 \
--motion_smooth_loss_weight 1e-3 \
--depth_consistency_loss_weight 1e-3 \
--rot_loss_weight 1e-3 \
--trans_loss_weight 1e-2 \
--use_norm_in_downsample

