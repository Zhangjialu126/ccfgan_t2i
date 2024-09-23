#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py \
--which_train_fn CFGAN \
--dataset I32 --parallel --shuffle  --num_workers 8 --batch_size 64 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 1e-4 --D_lr  2e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl leakyrelu --D_nl leakyrelu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--hier \
--G_init ortho --D_init ortho \
--dim_z 512 --shared_dim 1024 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--t_dim 256 \
--require_classifier \
--c_mode adc --CD_lambda 0.4 --CG_lambda 0.4 \
--which_best FID \
--test_every 2000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root './data' \
--name_suffix ty+adc0.4_z512_share1024 \
--num_epochs 3200 \
# --resume \