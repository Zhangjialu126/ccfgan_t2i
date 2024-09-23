''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import utils
import losses
import math
from ReACGAN_loss import Data2DataCrossEntropyLoss
import lib.datasets as t2i_dset
from models.inception import InceptionV3
import torchvision.transforms as transforms
import torch.distributed as dist
from lib.utils import get_rank
from tqdm import tqdm, trange
from lib.datasets import prepare_data
from lib.utils import transf_to_CLIP_input
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import inception_utils
import random


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}

    return train


def RCFGAN_training_function(G, D, GD, loss_fn, z_, y_, ema, state_dict, config, writer):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0
        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        reg = 6.0
        # reg = min(6.0 + 4 * state_dict['itr'] / 300000.0, 10.0)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                # D_fake, D_real, _, _ = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                #                                      x[counter], y[counter], train_G=False,
                #                                      split_D=config['split_D'])
                D_fake, D_real = GD(z_, y_, x=x[counter], train_G=False, split_D=config['split_D'])
                t_batch = D.net_t()
                recip_loss = nn.functional.mse_loss(D_fake, z_)
                gand_loss = loss_fn(t_batch, z_, D_real) - loss_fn(t_batch, z_, D_fake)
                # gand_loss = -loss_fn(t_batch, D_fake, D_real)
                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss = (gand_loss + reg * recip_loss) / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            D_fake, D_real = GD(z_, y_, x=x[counter], train_G=True, split_D=config['split_D'])
            t_batch = D.net_t()
            # NOTE
            # recip_loss = nn.functional.mse_loss(D_fake, z_)
            G_loss = loss_fn(t_batch.detach(), D_fake, D_real.detach()) / float(config['num_G_accumulations'])
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()) * 1e4,
               'D_loss': float(gand_loss.item()) * 1e4,
               'Recip_loss': float(recip_loss.item()) * 1e4}
        # Return G's loss and the components of D's loss.
        writer.add_scalar('G_loss', G_loss.item(), state_dict['itr'])
        writer.add_scalar('D_loss', gand_loss.item(), state_dict['itr'])
        writer.add_scalar('Recip_loss', recip_loss.item(), state_dict['itr'])
        return out

    return train


def CFGAN_training_function(G, D, GD, cf_loss_fn, z_, y_, ema, state_dict, config, writer):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step

            # train t_nets
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                cf_x, cf_target_batch = GD(z_, y_,
                                           x[counter], y[counter], train_G=False,
                                           split_D=config['split_D'])

                critic_loss = cf_loss_fn(cf_x, cf_target_batch)
                D_loss = - critic_loss
                D_loss = D_loss / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # training generator/sampler
        # train gan loss
        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            cf_x, cf_target_batch = GD(z_, y_,
                                       x[counter], y[counter], train_G=True,
                                       split_D=config['split_D'])
            # forward
            critic_loss = cf_loss_fn(cf_x, cf_target_batch)
            G_loss = critic_loss
            G_loss = G_loss / float(config['num_G_accumulations'])
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()) * 1e4,
               'D_loss': float(D_loss.item()) * 1e4}
        # Return G's loss and the components of D's loss.
        writer.add_scalar('G_loss', G_loss.item(), state_dict['itr'])
        writer.add_scalar('D_loss', D_loss.item(), state_dict['itr'])
        return out

    return train


def CFGAN_training_function_cond(G, D, image_encoder, text_encoder, GD, cf_loss_fn, z_, y_, ema, state_dict, config, writer):
    def train(x, caption, CLIP_token, key):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        # caption = torch.split(caption, config['batch_size'])
        CLIP_token = torch.split(CLIP_token, config['batch_size'])
        # key = torch.split(key, config['batch_size'])

        counter = 0
        if config['require_classifier'] == True:
            if config['ce_mode'] == 'CE':
                CELoss = nn.CrossEntropyLoss()
            elif config['ce_mode'] == 'D2DCE':
                CELoss = Data2DataCrossEntropyLoss(config['n_classes'])

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)
            utils.toggle_grad(image_encoder, False)
            utils.toggle_grad(text_encoder, False)


        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step

            # train t_nets
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                # y_.sample_()

                sent_emb, words_embs = t2i_dset.encode_tokens(text_encoder, CLIP_token[counter])
                # x = x.requires_grad_()
                sent_emb = sent_emb.requires_grad_()
                words_embs = words_embs.requires_grad_()

                # CLIP_real, real_emb = image_encoder(x[counter])

                cf_x, cf_target, pred_emb_x, pred_emb_target, G_z = GD(z_, x[counter], sent_emb, train_G=False, split_D=config['split_D'])

                critic_loss = cf_loss_fn(cf_x, cf_target)
                D_loss = - critic_loss / float(config['num_D_accumulations'])

                # emb loss
                mse_loss = nn.MSELoss()
                emb_loss_x = - torch.cosine_similarity(pred_emb_x, sent_emb.float().detach()).mean()
                emb_loss_target = - torch.cosine_similarity(pred_emb_target, sent_emb.float()).mean()
                # emb_loss_target = mse_loss(pred_emb_target, sent_emb.float().detach())
                emb_loss = (emb_loss_x + emb_loss_target) / float(config['num_D_accumulations'])

                # total loss
                total_loss = D_loss + emb_loss

                total_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)
            utils.toggle_grad(image_encoder, False)
            utils.toggle_grad(text_encoder, False)

        # training generator/sampler
        # train gan loss
        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            # y_.sample_()

            sent_emb, words_embs = t2i_dset.encode_tokens(text_encoder, CLIP_token[counter])
            # x = x.requires_grad_()
            sent_emb = sent_emb.requires_grad_()
            words_embs = words_embs.requires_grad_()

            cf_x, cf_target, pred_emb_x, pred_emb_target, G_z = GD(z_, x[counter], sent_emb, train_G=True, split_D=config['split_D'])

            # emb loss
            mse_loss = nn.MSELoss()
            emb_loss_x = - torch.cosine_similarity(pred_emb_x, sent_emb.float()).mean()
            emb_loss_target = - torch.cosine_similarity(pred_emb_target, sent_emb.float()).mean()
            # emb_loss_target = mse_loss(pred_emb_target, sent_emb.float())
            emb_loss = (emb_loss_x + emb_loss_target) / float(config['num_G_accumulations'])

            # text img sim
            CLIP_fake, fake_emb = image_encoder(G_z)
            text_img_sim = - torch.cosine_similarity(fake_emb, sent_emb.float()).mean() / float(config['num_G_accumulations'])

            # calculate critic loss
            critic_loss = cf_loss_fn(cf_x, cf_target)
            G_loss = critic_loss / float(config['num_G_accumulations'])

            # total loss
            total_loss = G_loss + emb_loss + text_img_sim

            total_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])

        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()) * 1e4,
               'D_loss': float(D_loss.item()) * 1e4,
               'emb_loss': float(emb_loss.item()) * 1e4}
        # Return G's loss and the components of D's loss.
        writer.add_scalar('G_loss', G_loss.item(), state_dict['itr'])
        writer.add_scalar('D_loss', D_loss.item(), state_dict['itr'])
        writer.add_scalar('emb_loss', emb_loss.item(), state_dict['itr'])
        return out

    return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config, writer):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                    x[counter], y[counter], train_G=False,
                                    split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        writer.add_scalar('G_loss', G_loss.item(), state_dict['itr'])
        writer.add_scalar('D_loss_real', D_loss_real.item(), state_dict['itr'])
        writer.add_scalar('D_loss_fake', D_loss_fake.item(), state_dict['itr'])
        return out

    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, fixed_img, fixed_sent, fixed_words,
                    state_dict, config, experiment_name, x=None):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z, fixed_sent))
        else:
            fixed_Gz = which_G(fixed_z, fixed_sent)
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    # utils.sample_sheet(which_G,
    #                    classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
    #                    num_classes=config['n_classes'],
    #                    samples_per_class=10, parallel=config['parallel'],
    #                    samples_root=config['samples_root'],
    #                    experiment_name=experiment_name,
    #                    folder_number=state_dict['itr'],
    #                    z_=z_)
    # Also save interp sheets
    # for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    #     utils.interp_sheet(which_G,
    #                        num_per_sheet=16,
    #                        num_midpoints=8,
    #                        num_classes=config['n_classes'],
    #                        parallel=config['parallel'],
    #                        samples_root=config['samples_root'],
    #                        experiment_name=experiment_name,
    #                        folder_number=state_dict['itr'],
    #                        sheet_number=0,
    #                        fix_z=fix_z, fix_y=fix_y, device='cuda')
    # if config['which_train_fn'] == 'RCFGAN':
    #     utils.interpolated_imgs(G, D, x, num_of_groups=10, config=config, device='cuda',
    #                             samples_root=config['samples_root'],
    #                             experiment_name=experiment_name,
    #                             folder_number=state_dict['itr'], )


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''


def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).sum()
    return text_img_sim


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# FID calculator from TTUR--consider replacing this with GPU-accelerated cov
# calculations using torch?
def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        print('wat')
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return out


def test(dataloader, text_encoder, G, D, G_ema, z_, CLIP, device, config, state_dict,
                               experiment_name, test_log, writer):
    print('Gathering inception metrics...')
    dataset = config['dataset'].strip('_hdf5')
    data_mu = np.load(dataset + '_inception_moments.npz')['mu']
    data_sigma = np.load(dataset + '_inception_moments.npz')['sigma']
    clip_cos = torch.FloatTensor([0.0]).to(device)
    inception_model = inception_utils.load_inception_net(config['parallel'])
    # inception_model = inception_model.to(device)
    # if config['parallel']:
    #     inception_model = nn.DataParallel(inception_model)
    # inception_model.eval()
    G.eval()
    pred_all = []
    generated_count = 0
    loop = tqdm(total=config['test_imgs_num'])

    while generated_count < config['test_imgs_num']:
        for data in dataloader[0]:
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            G.eval()
            with torch.no_grad():
                z_.sample_()
                if config['parallel']:
                    fake_imgs = nn.parallel.data_parallel(G, (z_, sent_emb))
                    fake_imgs = fake_imgs.float()
                else:
                    fake_imgs = G(z_, sent_emb)
                    fake_imgs = fake_imgs.float()
                pred, _ =inception_model(fake_imgs)
                # pred, _ = inception_model.module(fake_imgs) if config['parallel'] else inception_model(fake_imgs)
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim

                generated_count += config['batch_size']

                if generated_count < config['test_imgs_num']:
                    pred_all += [pred]
                else:
                    extra = config['test_imgs_num'] - generated_count
                    pred_all += [pred[:extra]]

            loop.update(config['batch_size'])
            if generated_count >= config['test_imgs_num']:
                break

    CLIP_score_gather = [torch.tensor([x]) for x in clip_cos]
    CLIP_score_concatenated = torch.cat(CLIP_score_gather, dim=0)
    clip_score = CLIP_score_concatenated.mean().item() / config['test_imgs_num']
    # FID
    pred_all = torch.cat(pred_all, 0)
    # mu, sigma = torch.mean(pred_all, 0), torch_cov(pred_all, rowvar=False)
    mu, sigma = np.mean(pred_all.cpu().numpy(), axis=0), np.cov(pred_all.cpu().numpy(), rowvar=False)
    # m2 = np.mean(pred_arr, axis=0)
    # s2 = np.cov(pred_arr, rowvar=False)
    # fid_value = inception_utils.torch_calculate_frechet_distance(mu, sigma, torch.tensor(data_mu).float().cuda(),
    #                                                              torch.tensor(data_sigma).float().cuda())
    # fid_value = float(fid_value.cpu().numpy())
    fid_value = numpy_calculate_frechet_distance(mu, sigma, data_mu, data_sigma)
    # fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print('Itr %d: FID is  is %5.4f, CLIP score is %5.4f' % (
        state_dict['itr'], fid_value, clip_score))
    # If improved over previous best metric, save approrpiate copy
    if fid_value < state_dict['best_FID']:
        print('%s improved over previous best, saving checkpoint...' % config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_CLIP'] = max(state_dict['best_CLIP'], clip_score)
    state_dict['best_FID'] = min(state_dict['best_FID'], fid_value)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), CLIP=float(clip_score), FID=float(fid_value))
    writer.add_scalar('FID', fid_value, state_dict['itr'])
    writer.add_scalar('CLIP', clip_score, state_dict['itr'])
# def test(dataloader, text_encoder, G, D, G_ema, z_, CLIP, device, m1, s1, config, state_dict,
#          experiment_name, test_log, writer):
#     print('Gathering inception metrics...')
#     clip_cos = torch.FloatTensor([0.0]).to(device)
#     inception_model = inception_utils.load_inception_net(config['parallel'])
#     inception_model =  inception_model.to(device)
#     if config['parallel']:
#         inception_model = nn.DataParallel(inception_model)
#     inception_model.eval()
#     G.eval()
#     pred_all = []
#     generated_count = 0
#     loop = tqdm(total=config['test_imgs_num'])
#
#     while generated_count < config['test_imgs_num']:
#         for data in dataloader[0]:
#             imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
#             G.eval()
#             with torch.no_grad():
#                 z_.sample_()
#                 if config['parallel']:
#                     fake_imgs = nn.parallel.data_parallel(G, (z_, sent_emb))
#                     fake_imgs = fake_imgs.float()
#                 else:
#                     fake_imgs = G(z_, sent_emb)
#                     fake_imgs = fake_imgs.float()
#                 pred, _ = inception_model.module(fake_imgs) if config['parallel'] else inception_model(fake_imgs)
#                 fake_imgs = torch.clamp(fake_imgs, -1., 1.)
#                 fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
#                 clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
#                 clip_cos = clip_cos + clip_sim
#
#                 generated_count += config['batch_size']
#
#                 if generated_count < config['test_imgs_num']:
#                     pred_all += [pred]
#                 else:
#                     extra = config['test_imgs_num'] - generated_count
#                     pred_all += [pred[:extra]]
#
#             loop.update(config['batch_size'])
#             if generated_count >= config['test_imgs_num']:
#                 break
#
#     CLIP_score_gather = [torch.tensor([x]) for x in clip_cos]
#     CLIP_score_concatenated = torch.cat(CLIP_score_gather, dim=0)
#     clip_score = CLIP_score_concatenated.mean().item() / config['test_imgs_num']
#     # FID
#     pred_all = torch.cat(pred_all, 0)
#     m2, s2 = torch.mean(pred_all, 0), torch_cov(pred_all, rowvar=False)
#     # m2 = np.mean(pred_arr, axis=0)
#     # s2 = np.cov(pred_arr, rowvar=False)
#     fid_value = inception_utils.torch_calculate_frechet_distance(m2, s2, torch.tensor(m1).float().cuda(),
#                                                                  torch.tensor(s1).float().cuda())
#     fid_value = float(fid_value.cpu().numpy())
#     # fid_value = calculate_frechet_distance(m1, s1, m2, s2)
#     print('Itr %d: FID is  is %5.4f, CLIP score is %5.4f' % (
#         state_dict['itr'], fid_value, clip_score))
#     # If improved over previous best metric, save approrpiate copy
#     if fid_value < state_dict['best_FID']:
#         print('%s improved over previous best, saving checkpoint...' % config['which_best'])
#         utils.save_weights(G, D, state_dict, config['weights_root'],
#                            experiment_name, 'best%d' % state_dict['save_best_num'],
#                            G_ema if config['ema'] else None)
#         state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
#     state_dict['best_CLIP'] = max(state_dict['best_CLIP'], clip_score)
#     state_dict['best_FID'] = min(state_dict['best_FID'], fid_value)
#     # Log results to file
#     test_log.log(itr=int(state_dict['itr']), CLIP=float(clip_score), FID=float(fid_value))
#     writer.add_scalar('FID', fid_value, state_dict['itr'])
#     writer.add_scalar('CLIP', clip_score, state_dict['itr'])


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
