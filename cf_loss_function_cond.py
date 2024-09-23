import torch
import torch.nn as nn
import os
import math


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


def calculate_ygx(ty, proj):  # calculate real and imaginary part of fai(y|x)
    proj = torch.unsqueeze(proj, -1)
    ty = torch.bmm(ty, proj)
    ty = torch.squeeze(ty, -1)
    return ty


def calculate_fai(out, ty):  # calculate real and imaginary part of fai(x,y)
    Re = torch.cos(out + ty)
    Im = torch.sin(out + ty)
    Re = torch.mean(Re, dim=0, keepdim=True)
    Im = torch.mean(Im, dim=0, keepdim=True)
    Norm = calculate_norm(Re, Im)
    return Re, Im, Norm


class CFLossFuncCond(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference
    Args:
        alpha: the weight for amplitude in CF loss, from 0-1
        beta: the weight for phase in CF loss, from 0-1

    """

    def __init__(self, alpha=0.5, beta=0.5):
        super(CFLossFuncCond, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, target, ty_x, ty_target, proj_x, proj_target):
        ty_x = calculate_ygx(ty_x, proj_x)
        ty_target = calculate_ygx(ty_target, proj_target)
        x_Re, x_Im, x_Norm = calculate_fai(x, ty_x)
        target_Re, target_Im, target_Norm = calculate_fai(target, ty_target)

        amp_diff = target_Norm - x_Norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (torch.mul(target_Norm, x_Norm) -
                        torch.mul(x_Re, target_Re) -
                        torch.mul(x_Im, target_Im))

        loss_amp = loss_amp.clamp(min=1e-12)  # keep numerical stability
        loss_pha = loss_pha.clamp(min=1e-12)  # keep numerical stability

        loss = torch.sqrt(torch.mean(self.alpha * loss_amp + self.beta * loss_pha))
        return loss
