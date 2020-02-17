import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class initLoss(nn.Module):
    def __init__(self, size_average=True):
        super(initLoss, self).__init__()
        # y = 1. / (1 + exp((x - 0.5). * a));
        self.alpha = 100.0 # 13: 200.0
        self.bias = 0.01  # 0.005
        self.wei0 = 1.0 / (1.0 + np.exp(-1.0 * self.bias * self.alpha))
        self.size_average = size_average
        return

    def forward(self, outS, outT, target):

        err = torch.add(outT, torch.mul(target, -1.0))
        err = torch.abs(err)

        wei = torch.mul(err - self.bias, self.alpha)
        wei = 1.0 + torch.exp(wei)
        wei = 1.0 / wei
        wei = wei / self.wei0

        out = torch.mul(outS, wei)
        target = torch.mul(target, wei)

        loss = func.mse_loss(out, target, size_average=self.size_average)

        return loss


class ailLoss(nn.Module):
    def __init__(self, size_average=True):
        super(ailLoss, self).__init__()
        self.alpha = 100.0 # 13: 200.0
        self.bias = 0.01
        self.wei0 = 1.0 / (1.0 + np.exp(-1.0 * self.bias * self.alpha))

        self.size_average = size_average
        self.relu = nn.ReLU(inplace=True)
        self.gamma = 0.1
        self.sigma = 10.0 # 10.0
        self.lam = 0.15 # 0.5 for SPL, 0.2 for SPC; 0.1 for each scale; 0.15 for all scales
        self.rho = 0.15 # SPM: 1.0; others: 1.3;  0.1 for each scale; 0.15 for all scales

        return

    def forward(self, outS, outS_p, outT, target, curr):

        err = torch.add(outT, torch.mul(target, -1.0))
        err = torch.abs(err)
        wei = torch.mul(err - self.bias, self.alpha)
        wei = 1.0 + torch.exp(wei)
        wei = 1.0 / wei
        wei = wei / self.wei0

        err2 = torch.add(outS_p, torch.mul(target, -1.0))
        err2 = torch.abs(err2)
        err2 = torch.mul(err2, -1.0 * self.sigma)
        delta = torch.exp(err2)
        lam = self.lam + self.rho * curr
        delta = torch.mul(delta, lam)

        wei = wei + delta
        wei = torch.clamp(wei, 0, 1.0)

        out = torch.mul(outS, wei)
        target = torch.mul(target, wei)

        loss = func.mse_loss(out, target, size_average=self.size_average) # This is for local desktop

        return loss
