import os
import time
import math


def stability_bash():
    arch = {'lr': [1e-4, 1e-3],
            'bs': [32, 64, 128],
            'ch': [64, 96],
            'GBlock': ['0_0_0', '1_1_1'],
            'DBlock': ['0_0_0_0', '1_1_1_1']}
    bash_ori = 'CUDA_VISIBLE_DEVICES=3 python train.py \
    --which_train_fn CFGAN \
    --dataset C10 --parallel --shuffle  --num_workers 8 \
    --num_G_accumulations 1 --num_D_accumulations 1 \
    --num_D_steps 1 --D_B2 0.999 --G_B2 0.999 \
    --G_attn 0 --D_attn 0 \
    --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
    --G_ortho 0.0 \
    --G_shared \
    --hier \
    --G_init N02 --D_init N02 \
    --dim_z 128 --shared_dim 128 \
    --G_eval_mode \
    --t_dim 256 \
    --require_classifier \
    --c_mode adc --CD_lambda 0.3 --CG_lambda 0.3 \
    --which_best FID \
    --test_every 1000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
    --data_root ./data '
    count = 0
    for lr in arch['lr']:
        for bs in arch['bs']:
            for ch in arch['ch']:
                for GBlock in arch['GBlock']:
                    for DBlock in arch['DBlock']:
                        if count > 5:
                            num_epochs = math.ceil(100000 * bs / 50000 * 2) + 1
                            with open("stability_process.txt", "a") as f:
                                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                x = 'lr: ' + str(lr) + ' bs: ' + str(bs) + ' ch: ' + str(ch) + ' GBlock: ' + str(
                                    GBlock) + ' DBlock: ' + str(DBlock) + '\n' \
                                    + 'start time: ' + str(start_time) + '\n'
                                f.write(x)
                            if count == 6:
                                os.system(str(bash_ori)
                                          + ' --G_lr ' + str(lr) + ' --D_lr ' + str(lr)
                                          + ' --batch_size ' + str(bs)
                                          + ' --G_ch ' + str(ch) + ' --D_ch ' + str(ch)
                                          + ' --G_block ' + str(GBlock)
                                          + ' --D_block ' + str(DBlock)
                                          + ' --num_epochs ' + str(num_epochs)
                                          + ' --resume')
                            else:
                                os.system(str(bash_ori)
                                          + ' --G_lr ' + str(lr) + ' --D_lr ' + str(lr)
                                          + ' --batch_size ' + str(bs)
                                          + ' --G_ch ' + str(ch) + ' --D_ch ' + str(ch)
                                          + ' --G_block ' + str(GBlock)
                                          + ' --D_block ' + str(DBlock)
                                          + ' --num_epochs ' + str(num_epochs))
                            with open("stability_process.txt", "a") as f:
                                end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                x = 'end time: ' + str(end_time) + '\n'
                                f.write(x)
                        count += 1


if __name__ == '__main__':
    stability_bash()