import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LAMB = 3
TRAIN = True


class Cconv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_lamb=NUM_LAMB, conv=True):
        super(Cconv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups,
                                      bias=bias, padding_mode=padding_mode)
        self.num_lamb = num_lamb
        self.channels_in = in_channels
        self.channels_out = out_channels
        self.with_conv = conv
        x = math.log(math.exp(1.) - 1)
        self.u = nn.Parameter(x * torch.ones([out_channels, 1, num_lamb]))
        self.v = nn.Parameter(torch.zeros([out_channels, 1, num_lamb]))

    def interp(self, x, lamb):
        map_list = [0,2,1]
        f1 = int(lamb)
        f2 = int(lamb + 1)
        f1 = map_list[f1]
        f2 = map_list[f2]
        ratio = lamb - np.floor(lamb)
        x_onehot_floor = F.one_hot(torch.LongTensor([f1]), num_classes=self.num_lamb)
        x_onehot_ceil = F.one_hot(torch.LongTensor([f2]), num_classes=self.num_lamb)
        x_onehot = (1. - ratio) * x_onehot_floor.type(dtype=torch.float32) + ratio * x_onehot_ceil.type(
            dtype=torch.float32)
        # print(x_onehot)
        x_onehot = x_onehot.repeat(self.channels_in, 1, 1).view(self.channels_in, -1, 1)
        return x_onehot.cuda()

    def forward(self, x, lamb, train=TRAIN):
        if self.in_channels == self.out_channels:
            if train:
                lamb = torch.LongTensor([lamb])
                x_onehot = F.one_hot(lamb, num_classes=self.num_lamb)
                x_onehot = x_onehot.type(dtype=torch.float32)
                x_onehot = x_onehot.repeat(self.channels_in, 1, 1).view(self.channels_in, -1, 1)
                x_onehot = x_onehot.cuda()
            else:
                x_onehot = self.interp(x, lamb)
            scale = torch.matmul(self.u, x_onehot)
            bias = torch.matmul(self.v, x_onehot)
            if self.with_conv:
                y = F.softplus(scale) * super(Cconv2d, self).forward(x) + bias
            else:
                y = F.softplus(scale) * x + bias
            return y
        else:
            return super(Cconv2d, self).forward(x)


class CconvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', num_lamb=NUM_LAMB, conv=True):
        super(CconvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                               padding=padding, output_padding=output_padding, groups=groups, bias=bias,
                                               dilation=dilation, padding_mode=padding_mode)

        self.num_lamb = num_lamb
        self.channels_in = in_channels
        self.channels_out = out_channels
        self.with_conv = conv
        x = math.log(math.exp(1.) - 1)
        self.u = nn.Parameter(x*torch.ones([out_channels, 1, num_lamb]))
        self.v = nn.Parameter(torch.zeros([out_channels, 1, num_lamb]))

    def interp(self, x, lamb):
        map_list = [0, 2, 1]
        f1 = int(lamb)
        f2 = int(lamb + 1)
        f1 = map_list[f1]
        f2 = map_list[f2]
        ratio = lamb - np.floor(lamb)
        x_onehot_floor = F.one_hot(torch.LongTensor([f1]), num_classes=self.num_lamb)
        x_onehot_ceil = F.one_hot(torch.LongTensor([f2]), num_classes=self.num_lamb)
        x_onehot = (1. - ratio) * x_onehot_floor.type(dtype=torch.float32) + ratio * x_onehot_ceil.type(
            dtype=torch.float32)
        x_onehot = x_onehot.repeat(self.channels_in, 1, 1).view(self.channels_in, -1, 1)
        return x_onehot.cuda()

    def forward(self, x, lamb, train=TRAIN):
        if self.in_channels == self.out_channels:
            if train:
                lamb = torch.LongTensor([lamb])
                x_onehot = F.one_hot(lamb, num_classes=self.num_lamb)
                x_onehot = x_onehot.type(dtype=torch.float32)
                x_onehot = x_onehot.repeat(self.channels_in, 1, 1).view(self.channels_in, -1, 1)
                x_onehot = x_onehot.cuda()
            else:
                x_onehot = self.interp(x, lamb)
            scale = torch.matmul(self.u, x_onehot)
            bias = torch.matmul(self.v, x_onehot)
            if self.with_conv:
                y = F.softplus(scale) * super(CconvTranspose2d, self).forward(x) + bias
            else:
                y = F.softplus(scale) * x + bias
            return y
        else:
            return super(CconvTranspose2d, self).forward(x)


class Csequential(nn.Sequential):
    def __init__(self, *args):
        super(Csequential, self).__init__(*args)

    def forward(self, input, lamb):
        for module in self:
            input = module(input, lamb)
        return input
