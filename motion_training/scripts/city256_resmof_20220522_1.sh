#!/bin/bash
python train.py \
--gpus_to_use 1 \
--data_path /seokju4/Dataset/fei-Dataset/cityscapes-Wild-256/train/ \
--png \
--weighted_ssim \
--boxify \
--model_name ./20220522_city256_softplus_resmof_smooth_sparsity_1/ \
--seg_mask none \
--log_frequency 2000 \
--save_frequency 2 \
--batch_size 12 \
--log_depth \
--log_lr \
--log_multiframe \
--log_trans whole \
--num_epochs 40 \
--reconstr_loss_weight 0.85 \
--ssim_loss_weight 1.5 \
--smooth_loss_weight 1e-3 \
--motion_smooth_loss_weight 1.5 \
--motion_sparsity_loss_weight 1.5 \
--depth_consistency_loss_weight 1e-2 \
--rot_loss_weight 1e-3 \
--trans_loss_weight 2e-2 \
--use_norm_in_downsample \
--epoch_only_ego 0 \
--load_weights_folder models/city256_softplus_ego/models/weights_9 \
--models_to_load encoder depth pose motion scaler