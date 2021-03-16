import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.uniform import Uniform

from Model.basic_module import Non_local_Block, ResBlock
from Model.context_model import P_Model
from Model.factorized_entropy_model import Entropy_bottleneck

import Util.util_variable as uv
from Model.gaussian_entropy_model import Distribution_for_entropy


class Disc(nn.Module):
    def __init__(self, M):
        super(Disc, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(M, 12, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(15, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, rec, fmap):
        fmap = self.conv1(fmap)
        conv2_in = torch.cat((rec, fmap), 1)
        output = self.conv2(conv2_in)
        return output



class PostResBlock(nn.Module):
    def __init__(self, in_channels_N):
        super(PostResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels_N, out_channels=in_channels_N, kernel_size=3, stride=1,
                               padding=1)
        # self.BN = nn.BatchNorm2d(num_features=in_channels_N)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels_N, out_channels=in_channels_N, kernel_size=3, stride=1,
                               padding=1)

    def forward(self, x):
        temp = self.conv1(x)
        # temp = self.BN(temp)
        temp = self.relu(temp)
        temp = self.conv2(temp)

        return x + temp


class PostProcNet(nn.Module):
    def __init__(self, in_channels_N=256):
        super(PostProcNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=in_channels_N, kernel_size=3, stride=1,
                               padding=1)

        self.relu = nn.ReLU()
        self.res = nn.Sequential(
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
            PostResBlock(in_channels_N=in_channels_N),
        )
        self.conv2 = nn.Conv2d(in_channels=in_channels_N, out_channels=in_channels_N, kernel_size=3, stride=1,
                               padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=in_channels_N, out_channels=3, kernel_size=3, stride=1,
                               padding=1)

    def forward(self, x):
        temp = self.conv1(x)
        temp = self.relu(temp)
        temp = self.res(temp)
        temp = self.conv2(temp)
        temp = self.relu(temp)
        temp = self.conv3(temp)
        return x + temp

class Enc(nn.Module):
    def __init__(self, num_features, N1, N2, M, M1):
        # input_features = 3, N1 = 192, N2 = 128, M = 192, M1 = 96
        super(Enc, self).__init__()
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.M = int(M)
        self.M1 = int(M1)
        self.n_features = int(num_features)

        self.conv1 = uv.Cconv2d(self.n_features, self.M1, 5, 1, 2)
        self.trunk1 = uv.Csequential(ResBlock(self.M1, self.M1, 3, 1, 1), ResBlock(
            self.M1, self.M1, 3, 1, 1), uv.Cconv2d(self.M1, 2 * self.M1, 5, 2, 2))

        self.down1 = uv.Cconv2d(2 * self.M1, self.M, 5, 2, 2)
        self.trunk2 = uv.Csequential(ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                     ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                     ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1))
        self.mask1 = uv.Csequential(Non_local_Block(2 * self.M1, self.M1), ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(
                                        2 * self.M1, 2 * self.M1, 3, 1, 1), ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    uv.Cconv2d(2 * self.M1, 2 * self.M1, 1, 1, 0))
        self.trunk3 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1), uv.Cconv2d(self.M, self.M, 5, 2, 2))

        self.trunk4 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1), uv.Cconv2d(self.M, self.M, 5, 2, 2))

        self.trunk5 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1))
        self.mask2 = uv.Csequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), uv.Cconv2d(self.M, self.M, 1, 1, 0))

        # hyper

        self.trunk6 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     uv.Cconv2d(self.M, self.M, 5, 2, 2))
        self.trunk7 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     uv.Cconv2d(self.M, self.M, 5, 2, 2))

        self.trunk8 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1))
        self.mask3 = uv.Csequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), uv.Cconv2d(self.M, self.M, 1, 1, 0))
        self.conv2 = uv.Cconv2d(self.M, self.N2, 3, 1, 1)

    def forward(self, x, lamb):
        x1 = self.conv1(x, lamb)
        x2 = self.trunk1(x1, lamb)
        x3 = self.trunk2(x2, lamb) + x2
        x3 = self.down1(x3, lamb)
        x4 = self.trunk3(x3, lamb)
        x5 = self.trunk4(x4, lamb)
        x6 = self.trunk5(x5, lamb) * f.sigmoid(self.mask2(x5, lamb)) + x5
        # hyper
        x7 = self.trunk6(x6, lamb)
        x8 = self.trunk7(x7, lamb)
        x9 = self.trunk8(x8, lamb) * f.sigmoid(self.mask3(x8, lamb)) + x8
        x10 = self.conv2(x9, lamb)

        return x6, x10


class Hyper_Dec(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Dec, self).__init__()

        self.N2 = N2
        self.M = M
        self.conv1 = uv.Cconv2d(self.N2, M, 3, 1, 1)
        self.trunk1 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1))
        self.mask1 = uv.Csequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), uv.Cconv2d(self.M, self.M, 1, 1, 0))

        self.trunk2 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     uv.CconvTranspose2d(M, M, 5, 2, 2, 1))
        self.trunk3 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     uv.CconvTranspose2d(M, M, 5, 2, 2, 1))

    def forward(self, xq2, lamb):
        x1 = self.conv1(xq2, lamb)
        x2 = self.trunk1(x1, lamb) * f.sigmoid(self.mask1(x1, lamb)) + x1
        x3 = self.trunk2(x2, lamb)
        x4 = self.trunk3(x3, lamb)
        return x4


class Dec(nn.Module):
    def __init__(self, input_features, N1, M, M1):
        super(Dec, self).__init__()

        self.N1 = N1
        self.M = M
        self.M1 = M1
        self.input = input_features

        self.trunk1 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1))
        self.mask1 = uv.Csequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), uv.Cconv2d(self.M, self.M, 1, 1, 0))

        self.up1 = uv.CconvTranspose2d(M, M, 5, 2, 2, 1)
        self.trunk2 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1), uv.CconvTranspose2d(M, M, 5, 2, 2, 1))
        self.trunk3 = uv.Csequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                     ResBlock(self.M, self.M, 3, 1, 1), uv.CconvTranspose2d(M, 2 * self.M1, 5, 2, 2, 1))

        self.trunk4 = uv.Csequential(ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                     ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                     ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1))
        self.mask2 = uv.Csequential(Non_local_Block(2 * self.M1, self.M1), ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    uv.Cconv2d(2 * self.M1, 2 * self.M1, 1, 1, 0))

        self.trunk5 = uv.Csequential(uv.CconvTranspose2d(2 * M1, M1, 5, 2, 2, 1), ResBlock(self.M1, self.M1, 3, 1, 1),
                                     ResBlock(self.M1, self.M1, 3, 1, 1),
                                     ResBlock(self.M1, self.M1, 3, 1, 1))

        self.conv1 = uv.Cconv2d(self.M1, self.input, 5, 1, 2)

    def forward(self, x, lamb):
        x1 = self.trunk1(x, lamb) * f.sigmoid(self.mask1(x, lamb)) + x
        x1 = self.up1(x1, lamb)
        x2 = self.trunk2(x1, lamb)
        x3 = self.trunk3(x2, lamb)
        x4 = self.trunk4(x3, lamb) + x3
        # print (x4.size())
        x5 = self.trunk5(x4, lamb)
        output = self.conv1(x5, lamb)
        return output


class Image_coding(nn.Module):
    def __init__(self, input_features, N1, N2, M, M1):
        # input_features = 3, N1 = 192, N2 = 128, M = 192, M1 = 96
        super(Image_coding, self).__init__()
        self.N1 = N1
        self.encoder = Enc(input_features, N1, N2, M, M1)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.p = P_Model(M)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.decoder = Dec(input_features, N1, M, M1)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, if_training, lamb):
        x1, x2 = self.encoder(x, lamb)
        xq2, xp2 = self.factorized_entropy_func(x2, if_training)
        x3 = self.hyper_dec(xq2, lamb)
        hyper_dec = self.p(x3, lamb)
        if if_training == 0:
            xq1 = self.add_noise(x1)
        elif if_training == 1:
            xq1 = UniverseQuant.apply(x1)
        else:
            xq1 = torch.round(x1)
        xp1 = self.gaussin_entropy_func(xq1, hyper_dec)
        output = self.decoder(xq1, lamb)

        return [output, xp1, xp2, xq1, hyper_dec]


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = np.random.uniform(-1, 1)
        # b = 0
        uniform_distribution = Uniform(-0.5 * torch.ones(x.size())
                                       * (2 ** b), 0.5 * torch.ones(x.size()) * (2 ** b)).sample().cuda()
        return torch.round(x + uniform_distribution) - uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g