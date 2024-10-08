""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
from loss_function import CFLossFunc
from cf_loss_function import CFLossFunc2
from cf_loss_function_cond import CFLossFuncCond
from torch.utils.tensorboard import SummaryWriter
import clip
import importlib
from lib.datasets import get_fix_data
from lib.utils import load_npz


def load_clip(clip_info, device):
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    # config['n_classes'] = 500
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'
    # config['n_classes'] = 1  # Single label usage for unconditional GAN

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True
    assert (config['hier'] == config['G_shared']), 'Hier and GShared should be both true or false.'

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    CLIP4trn = load_clip({'src':"clip", 'type':'ViT-B/32'} , device).eval()
    CLIP4evl = load_clip({'src':"clip", 'type':'ViT-B/32'} , device).eval()
    CLIP_img_enc = model.CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    image_encoder = CLIP_img_enc.eval()
    CLIP_txt_enc = model.CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    text_encoder = CLIP_txt_enc.eval()
    G = model.Generator(CLIP4trn, **config).to(device)
    D = model.Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(CLIP4trn, **{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GD = model.G_D(G, D, image_encoder, text_encoder, **config)
    print(G)
    print(D)
    # print(image_encoder)
    # print(text_encoder)
    print('Number of params in G: {} D: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'best_CLIP': 0, 'best_FID': 999999,
                    'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, state_dict, config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None, G_ema if config['ema'] else None)

    # If parallel, parallelize the GD module
    if config['parallel']:
        GD = nn.DataParallel(GD)
        if config['cross_replica']:
            patch_replication_callback(GD)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.json' % (config['logs_root'],
                                             experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    tbwriter = SummaryWriter('%s/%s/' % (config['logs_root'], experiment_name))
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    if config['which_train_fn'] == 'RCFGAN':
        D_batch_size = (config['batch_size'] * (
                    config['num_D_accumulations'] * config['num_D_steps'] + config['num_G_accumulations']))
    elif config['which_train_fn'] == 'CFGAN':
        D_batch_size = (config['batch_size'] * (
                    config['num_D_accumulations'] * config['num_D_steps'] + config['num_G_accumulations']))
    else:
        D_batch_size = (config['batch_size'] * (
                config['num_D_accumulations'] * config['num_D_steps']))
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    train_loaders, test_loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size, 'test_batch_size': G_batch_size,
                                                            'start_itr': state_dict['itr']})
    # 假设 train_loaders 是一个包含多个 DataLoader 的列表
    # for i, dataloader in enumerate(train_loaders):
    #     print(f'Total samples in train dataset {i}: {len(dataloader.dataset)}')
    # for i, dataloader in enumerate(test_loaders):
    #     print(f'Total samples in test dataset {i}: {len(dataloader.dataset)}')

    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'],
                                                                      config['no_fid'])

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    assert G_batch_size == config['batch_size'], 'Currently we do not allow different bs for G!'
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], z_var=config['z_var'],
                               device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_img, fixed_sent, fixed_words = get_fix_data(train_loaders, test_loaders, text_encoder, device, G_batch_size)
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], z_var=config['z_var'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    # # Load m1 and s1 for test
    # m1, s1 = load_npz(config['npz_path'])

    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                                ema, state_dict, config, tbwriter)
    elif config['which_train_fn'] == 'RCFGAN':
        loss_fn = CFLossFunc()
        train = train_fns.RCFGAN_training_function(G, D, GD, loss_fn, z_, y_,
                                                   ema, state_dict, config, tbwriter)
    elif config['which_train_fn'] == 'CFGAN':
        if config['unconditional']:
            cf_loss_fn = CFLossFunc2()
            train = train_fns.CFGAN_training_function(G, D, GD, cf_loss_fn, z_, y_,
                                                        ema, state_dict, config, tbwriter)
        else:
            cf_loss_fn = CFLossFunc2()
            # cf_loss_fn = CFLossFuncCond()
            train = train_fns.CFGAN_training_function_cond(G, D, image_encoder, text_encoder, GD, cf_loss_fn, z_, y_,
                                                        ema, state_dict, config, tbwriter)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                  else G),
                               z_=z_, y_=y_, config=config)

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(train_loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(train_loaders[0])
        for i, (x, caption, CLIP_token, key) in enumerate(pbar):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, CLIP_token = x.to(device).half(), CLIP_token.to(device)
            else:
                x, CLIP_token = x.to(device), CLIP_token.to(device)
            metrics = train(x, caption, CLIP_token, key)
            train_log.log(itr=int(state_dict['itr']), **metrics)

            # # Since RCF-GAN is quite stable, we donot record those values to save spaces
            # # Every sv_log_interval, log singular values
            # if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
            #     train_log.log(itr=int(state_dict['itr']),
            #                   **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']]
                                + ['%s : %+4.3f' % (key, metrics[key])
                                   for key in metrics]), end=' ')

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, fixed_img, fixed_sent, fixed_words,
                                          state_dict, config, experiment_name, x=x)

            # Test every specified interval
            if not (state_dict['itr'] % config['test_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                train_fns.test(test_loaders, text_encoder, G, D, G_ema, z_, CLIP4evl, device, config,
                               state_dict, experiment_name, test_log, tbwriter)
                # train_fns.test(test_loaders, text_encoder, G, D, G_ema, z_, CLIP4evl, device, m1, s1, config,
                #                state_dict, experiment_name, test_log, tbwriter)
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()