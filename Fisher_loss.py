import torch
import torch.nn as nn
import os
import math
from torch.autograd import Variable


class FisherLoss(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference
    Args:
        alpha: the weight for amplitude in CF loss, from 0-1
        beta: the weight for phase in CF loss, from 0-1

    """

    def __init__(self):
        super(FisherLoss, self).__init__()
        # # Initialize alpha
        # self.LAMBDA = Variable(torch.zeros(1), requires_grad=True).to('cuda')
        # self.RHO = Variable(torch.tensor(RHO), requires_grad=True).to('cuda')

    def forward(self, D_fake, D_real, LAMBDA, RHO):
        # First and second order central moments (Gaussian assumed)
        D_fake_moment_1, D_real_moment_1 = D_fake.mean(), D_real.mean()
        D_fake_moment_2, D_real_moment_2 = (D_fake ** 2).mean(), (D_real ** 2).mean()

        # Compute constraint on second order moments
        OMEGA = 1 - (0.5 * D_fake_moment_2 + 0.5 * D_real_moment_2)

        # Compute loss (Eqn. 9)
        D_loss = -((D_real_moment_1 - D_fake_moment_1) + LAMBDA * OMEGA - (RHO / 2) * (OMEGA ** 2))

        # For progress logging
        IPM_enum = D_real_moment_1.item() - D_fake_moment_1.item()
        IPM_denom = (0.5 * D_real_moment_2.item() + 0.5 * D_fake_moment_2.item()) ** 0.5
        IPM_ratio = IPM_enum / IPM_denom

        return D_loss, IPM_ratio