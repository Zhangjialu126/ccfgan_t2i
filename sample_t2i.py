''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import os
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
from pytorch_fid.fid_score import calculate_fid_given_paths
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import datasets as dset
from utils import CenterCropLongEdge
import clip
import lib.datasets as t2i_dset
from lib.datasets import prepare_data
from lib.datasets import get_one_batch_data
from lib.utils import transf_to_CLIP_input


def load_clip(clip_info, device):
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model


def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim


def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_CLIP': 0, 'best_FID': 999999, 'config': config}

    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    if config['config_from_name']:
        utils.load_weights(None, None, state_dict, config['weights_root'],
                           config['experiment_name'], config['load_weights'], None,
                           strict=False, load_optim=False)
        # Ignore items which we might want to overwrite from the command line
        for item in state_dict['config']:
            if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
                config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    if config['recons']:
        D = model.Discriminator(**config).cuda()
    CLIP4trn = load_clip({'src': "clip", 'type': 'ViT-B/32'}, device).eval()
    CLIP4evl = load_clip({'src': "clip", 'type': 'ViT-B/32'}, device).eval()
    CLIP_img_enc = model.CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    image_encoder = CLIP_img_enc.eval()
    CLIP_txt_enc = model.CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    text_encoder = CLIP_txt_enc.eval()
    G = model.Generator(CLIP4evl, **config).cuda()
    utils.count_parameters(G)

    # Load weights
    print('Loading weights...')
    # Here is where we deal with the ema--load ema weights or load normal weights
    utils.load_weights(G if not (config['use_ema']) else None, D if config['recons'] else None, state_dict,
                       config['weights_root'], experiment_name, config['load_weights'],
                       G if config['ema'] and config['use_ema'] else None,
                       strict=False, load_optim=False)
    # Update batch size setting used for G
    G_batch_size = config['batch_size']  # max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], z_var=config['z_var'],
                               device=device, fp16=config['G_fp16'])

    if config['G_eval_mode']:
        print('Putting G in eval mode..')
        G.eval()
    else:
        print('G is in %s mode...' % ('training' if G.training else 'eval'))

    # Sample a number of images and save them to an NPZ, for use with TF-Inception
    if config['sample_npz']:
        # convinient root dataset
        root_dict = {'I32': 'ImageNet', 'I32_hdf5': 'ILSVRC32.hdf5',
                     'I64': 'ImageNet', 'I64_hdf5': 'ILSVRC64.hdf5',
                     'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
                     'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
                     'C10': 'cifar', 'C100': 'cifar',
                     'L64': 'lsun', 'L128': 'lsun', 'A64': 'celeba',
                     'A128': 'celeba', 'I_tiny': 'tiny-imagenet-200', 'VGGFace2': 'VGG-Face2',
                     'V200': 'VGG-Face200', 'V500': 'VGG-Face500', 'V1000': 'VGG-Face1000',
                     'CUB200': 'CUB200', 'SVHN': 'SVHN',
                     'CUB128': 'CUB200', 't2i_birds': 'birds'}
        dset_dict = {'I32': dset.ImageFolder, 'I64': dset.ImageFolder,
                     'I128': dset.ImageFolder, 'I256': dset.ImageFolder,
                     'I32_hdf5': dset.ILSVRC_HDF5, 'I64_hdf5': dset.ILSVRC_HDF5,
                     'I128_hdf5': dset.ILSVRC_HDF5, 'I256_hdf5': dset.ILSVRC_HDF5,
                     'C10': dset.CIFAR10, 'C100': dset.CIFAR100,
                     'L64': torchvision.datasets.LSUN, 'L128': torchvision.datasets.LSUN,
                     'A64': dset.ImageFolder, 'A128': dset.ImageFolder,
                     'I_tiny': dset.ImageFolder, 'VGGFace2': dset.ImageFolder,
                     'V200': dset.ImageFolder, 'V500': dset.ImageFolder, 'V1000': dset.ImageFolder,
                     'CUB200': dset.ImageFolder, 'SVHN': torchvision.datasets.SVHN,
                     'CUB128': dset.ImageFolder, 't2i_birds': t2i_dset.TextImgDataset}
        imsize_dict = {'I32': 32, 'I32_hdf5': 32,
                       'I64': 64, 'I64_hdf5': 64,
                       'I128': 128, 'I128_hdf5': 128,
                       'I256': 256, 'I256_hdf5': 256,
                       'C10': 32, 'C100': 32,
                       'L64': 64, 'L128': 128, 'A64': 64,
                       'A128': 128, 'I_tiny': 64, 'VGGFace2': 64,
                       'V200': 64, 'V500': 64, 'V1000': 64,
                       'CUB200': 64, 'SVHN': 32,
                       'CUB128': 128, 't2i_birds': 256}
        # We first sample real images
        x = []
        print('Sampling %d generated images and saving them to npz...' % config['sample_num_npz'])
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        data_root = config['data_root'] + '/' + root_dict[config['dataset']]
        which_dataset = dset_dict[config['dataset']]
        sample_transform = transforms.Compose([
            transforms.Resize(int(imsize_dict[config['dataset']] * 76 / 64)),
            transforms.RandomCrop(imsize_dict[config['dataset']]),
            transforms.RandomHorizontalFlip(),
            ])
        t2i_kwargs = {'clip4text': {'src': "clip", 'type': 'ViT-B/32'}, 'data_dir': data_root, 'dataset_name': 'birds'}
        sample_set = which_dataset(split='test', transform=sample_transform, args=t2i_kwargs)
        sample_loader = DataLoader(sample_set, batch_size=G_batch_size,
                                   shuffle=True, num_workers=16)
        for i, data in enumerate(sample_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            print(i)
            images, _, _, _ = data
            x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        x = x.transpose((0, 2, 3, 1))
        real_npz_filename = '%s/%s/reals.npz' % (config['samples_root'], experiment_name)
        print('Saving npz to %s...' % real_npz_filename)
        np.savez(real_npz_filename, **{'x': x})

        # Now we sample fake images
        x = []
        print('Sampling %d generated images and saving them to npz...' % config['sample_num_npz'])
        generated_count = 0
        clip_cos = torch.FloatTensor([0.0]).to(device)
        loop = tqdm(total=config['sample_num_npz'])
        while generated_count < config['sample_num_npz']:
            for data in sample_loader:
                imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
                current_batch_size = sent_emb.size(0)
                with torch.no_grad():
                    z_.sample_()
                    if current_batch_size < G_batch_size:
                        z = z_[:current_batch_size, :]
                    else:
                        z = z_
                    if config['parallel']:
                        images = nn.parallel.data_parallel(G, (z, sent_emb))
                    else:
                        images = G(z, sent_emb)
                    x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
                    fake_imgs = images.float()
                    fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                    fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                    clip_sim = calc_clip_sim(CLIP4evl, fake_imgs, CLIP_tokens, device)
                    clip_cos = clip_cos + clip_sim
                generated_count += G_batch_size
                loop.update(G_batch_size)
                if generated_count >= config['sample_num_npz']:
                    break
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        x = x.transpose((0, 2, 3, 1))
        fake_npz_filename = '%s/%s/samples.npz' % (config['samples_root'], experiment_name)
        print('Saving npz to %s...' % fake_npz_filename)
        np.savez(fake_npz_filename, **{'x': x})

        CLIP_score_gather = [torch.tensor([x]) for x in clip_cos]
        CLIP_score_concatenated = torch.cat(CLIP_score_gather, dim=0)
        clip_score = CLIP_score_concatenated.mean().item() / config['sample_num_npz']

        # calculate tf FIDs
        paths = []
        paths += [real_npz_filename]
        paths += [fake_npz_filename]
        print('Now calculating fids ...')
        tf_fid = calculate_fid_given_paths(paths, batch_size=500)
        print('Pytorch FID is: ', tf_fid)
        print('CLIP score is: ', clip_score)
        metircs_filename = '%s/%s/%d/metrics.txt' % (config['samples_root'], experiment_name, config['sample_sheet_folder_num'])
        with open(metircs_filename, 'w', encoding='utf-8') as f:
            f.write(f"Pytorch FID is:\t{tf_fid}\n")
            f.write(f"CLIP score is:\t{clip_score}\n")

    # Save ground truth image, generated image and the corresponding text
    with torch.no_grad():
        z_.sample_()
        real_imgs, captions, CLIP_tokens, sent_emb, words_embs, keys= get_one_batch_data(sample_loader, text_encoder, device, G_batch_size)
        if config['parallel']:
            fake_imgs = nn.parallel.data_parallel(G, (z_, sent_emb))
        else:
            fake_imgs = G(z_, sent_emb)
        if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
            os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
        if not os.path.isdir('%s/%s/%d' % (config['samples_root'], experiment_name, config['sample_sheet_folder_num'])):
            os.mkdir('%s/%s/%d' % (config['samples_root'], experiment_name, config['sample_sheet_folder_num']))
        image_filename_fake = '%s/%s/%d/samples_fake.jpg' % (config['samples_root'], experiment_name,
                                                             config['sample_sheet_folder_num'])
        torchvision.utils.save_image(fake_imgs.float().cpu(), image_filename_fake,
                                     nrow=int(fake_imgs.shape[0] ** 0.5), normalize=True)
        image_filename_real = '%s/%s/%d/samples_real.jpg' % (config['samples_root'], experiment_name,
                                                             config['sample_sheet_folder_num'])
        torchvision.utils.save_image(real_imgs.float().cpu(), image_filename_real,
                                     nrow=int(real_imgs.shape[0] ** 0.5), normalize=True)
        captions_filename = '%s/%s/%d/text.txt' % (config['samples_root'], experiment_name, config['sample_sheet_folder_num'])
        with open(captions_filename, 'w', encoding='utf-8') as f:
            for i, caption in enumerate(captions):
                f.write(f"{i}\t{caption}\n")

    # # Sample interp sheets
    # if config['sample_interps']:
    #     print('Preparing interp sheets...')
    #     for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    #         utils.interp_sheet(G, num_per_sheet=16, num_midpoints=8,
    #                            num_classes=config['n_classes'],
    #                            parallel=config['parallel'],
    #                            samples_root=config['samples_root'],
    #                            experiment_name=experiment_name,
    #                            folder_number=config['sample_sheet_folder_num'],
    #                            sheet_number=0,
    #                            fix_z=fix_z, fix_y=fix_y, device='cuda')


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
