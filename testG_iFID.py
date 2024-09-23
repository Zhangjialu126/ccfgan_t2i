''' Test
   This script loads a pretrained net and a weightsfile and test '''
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
import datasets as dset

from sklearn.linear_model import LogisticRegression

import torchvision.transforms as transforms
from utils import CenterCropLongEdge
from torch.utils.data import DataLoader
from pytorch_fid.fid_score import calculate_fid_given_paths

from torch.utils.data.dataset import Subset


def testG_iFID(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

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
    config['recons'] = None
    config['sample_num_npz'] = 5000
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

    G = model.Generator(**config).cuda()
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
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'],
                               z_var=config['z_var'])

    if config['G_eval_mode']:
        print('Putting G in eval mode..')
        G.eval()
    else:
        print('G is in %s mode...' % ('training' if G.training else 'eval'))

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
                 'CUB128': 'CUB200'}
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
                 'CUB128': dset.ImageFolder}
    imsize_dict = {'I32': 32, 'I32_hdf5': 32,
                   'I64': 64, 'I64_hdf5': 64,
                   'I128': 128, 'I128_hdf5': 128,
                   'I256': 256, 'I256_hdf5': 256,
                   'C10': 32, 'C100': 32,
                   'L64': 64, 'L128': 128, 'A64': 64,
                   'A128': 128, 'I_tiny': 64, 'VGGFace2': 64,
                   'V200': 64, 'V500': 64, 'V1000': 64,
                   'CUB200': 64, 'SVHN': 32,
                   'CUB128': 128}

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    data_root = config['data_root'] + '/' + root_dict[config['dataset']]
    which_dataset = dset_dict[config['dataset']]
    if config['dataset'] in ['C10', 'C100']:
        sample_transform = []
    elif config['dataset'] in ['L64', 'L128', 'A64', 'A128']:
        sample_transform = [transforms.Resize(imsize_dict[config['dataset']]),
                            transforms.CenterCrop(imsize_dict[config['dataset']])]
        if config['dataset'] in ['L64', 'L128']:
            dataset_kwargs = {'classes': ['bedroom_train']}
    else:
        sample_transform = [CenterCropLongEdge(), transforms.Resize(imsize_dict[config['dataset']])]
    sample_transform = transforms.Compose(sample_transform + [
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])
    dataset_kwargs = {'index_filename': '%s_imgs.npz' % config['dataset']}
    if config['dataset'] in ['SVHN']:
        sample_set = which_dataset(root=data_root, transform=sample_transform, download=True)
    else:
        sample_set = which_dataset(root=data_root, transform=sample_transform, **dataset_kwargs)
    ## too slow!!
    # indices = [[] for _ in range(utils.nclass_dict[config['dataset']])]
    # for i in tqdm(range(len(sample_set))):
    #     for label in range(utils.nclass_dict[config['dataset']]):
    #         if int(sample_set[i][1]) == label:
    #             indices[label].append(i)
    ## change 1:
    # indices = torch.zeros(len(sample_set), utils.nclass_dict[config['dataset']], dtype=torch.bool)
    # for i in tqdm(range(len(sample_set)), desc="Subset by label"):
    #     label = int(sample_set[i][1])
    #     indices[i, label] = 1
    ## change 2:
    indices = [[] for _ in range(utils.nclass_dict[config['dataset']])]
    for i in tqdm(range(len(sample_set)), desc="Subset by label"):
        label = int(sample_set[i][1])
        indices[label].append(i)
    FIDs = []
    for label in range(utils.nclass_dict[config['dataset']]):
        # We first sample real images
        x = []

        ## too slow!!!
        # indices = []
        # for i in range(len(sample_set)):
        #     if int(sample_set[i][1]) == label:
        #         indices.append(i)
        # for i in tqdm(range(len(sample_set))):
        #     if int(sample_set[i][1]) == label:
        #         indices.append(i)
        idx = indices[label]
        sample_set_new = Subset(sample_set, [i for i in idx])
        ## change 1:
        # indices = [i for i in range(len(sample_set)) if int(sample_set[i][1]) == label]
        # indices = torch.tensor(indices, device='cuda')
        # sample_set = Subset(sample_set, indices)
        ## change 2:
        # sample_tensor = torch.tensor(sample_set)
        # labels = sample_tensor[:, 1]
        # indices = torch.where(labels == label)[0]
        # idx = indices.tolist()
        # sample_set = Subset(sample_set, idx)

        sample_loader = DataLoader(sample_set_new, batch_size=G_batch_size,
                                   shuffle=True, num_workers=16)
        config['sample_num_npz'] = len(sample_set_new)
        print('Sampling %d generated images of label %d and saving them to npz...' % (config['sample_num_npz'], label))
        for i, data in enumerate(sample_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # print(i)
            images, _ = data
            x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
            if i > int(np.ceil(config['sample_num_npz'] / float(G_batch_size))):
                break
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        x = x.transpose((0, 2, 3, 1))
        real_npz_filename = '%s/%s/%d_reals.npz' % (config['samples_root'], experiment_name, label)
        print('Saving npz to %s...' % real_npz_filename)
        np.savez(real_npz_filename, **{'x': x})

        # Now we sample fake images
        z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                                   device=device, fp16=config['G_fp16'], label=label)
        sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
        x = []
        print('Sampling %d generated images and saving them to npz...' % config['sample_num_npz'])
        for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
            with torch.no_grad():
                images, _ = sample()
                x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        x = x.transpose((0, 2, 3, 1))
        fake_npz_filename = '%s/%s/%d_samples.npz' % (config['samples_root'], experiment_name, label)
        print('Saving npz to %s...' % fake_npz_filename)
        np.savez(fake_npz_filename, **{'x': x})

        # calculate tf FIDs
        paths = []
        paths += [real_npz_filename]
        paths += [fake_npz_filename]
        print('Now calculating fid of label %d ...' % label)
        tf_fid = calculate_fid_given_paths(paths, batch_size=500)
        print('Pytorch FID of label %d is: ' % label, tf_fid)

        # # Prepare inception metrics: FID and IS
        # get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'],
        #                                                                   config['no_fid'], label=label)
        # z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
        #                            device=device, fp16=config['G_fp16'], label=label)
        # sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
        # IS_mean, IS_std, FID = get_inception_metrics(sample,
        #                                              config['num_inception_images'],
        #                                              num_splits=10)
        print(tf_fid)
        FIDs.append(tf_fid)
        filename = '%s/%s/intraFID.txt' % (config['samples_root'], experiment_name)
        with open(filename, "a") as files:
            files.write(f"\nIntraFID of label {label} is: {tf_fid}")
    print(np.mean(FIDs))
    filename = '%s/%s/intraFID.txt' % (config['samples_root'], experiment_name)
    with open(filename, "a") as files:
        files.write(f"\nMean IntraFID of is: {np.mean(FIDs)}")


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    # parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    testG_iFID(config)


if __name__ == '__main__':
    main()