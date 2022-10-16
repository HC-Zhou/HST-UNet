# -*- coding:utf-8  -*-
"""
Time: 2022/9/13 8:21
Author: Yimohanser
Software: PyCharm
"""
import torch
from torch import nn
from collections import OrderedDict
from lib.sseg.HST.SSA import shunted_s


class LWF(nn.Module):
    def __init__(self, dims=128, eps=1e-8):
        super(LWF, self).__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.post_conv = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU()
        )

    def forward(self, x, skip):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * skip + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2,
                                          output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.deconv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x


class HST_UNet_LWF(nn.Module):
    def __init__(self, img_size=256, pretrained='', num_classes=1, vis=None):
        super(HST_UNet_LWF, self).__init__()
        self.encoder = shunted_s(img_size=img_size, pretrained=pretrained)
        self.WF4 = LWF(256)
        self.decoder4 = UpSampleBlock(in_channels=512, out_channels=256)

        self.WF3 = LWF(128)
        self.decoder3 = UpSampleBlock(in_channels=256, out_channels=128)

        self.WF2 = LWF(64)
        self.decoder2 = UpSampleBlock(in_channels=128, out_channels=64)

        self.decoder1 = UpSampleBlock(in_channels=64, out_channels=32)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1, bias=False),
        )

    def forward(self, x):
        result = OrderedDict()
        e1, e2, e3, e4 = self.encoder(x)
        d4 = self.decoder4(e4)
        d4 = self.WF4(d4, e3)

        d3 = self.decoder3(d4)
        d3 = self.WF3(d3, e2)

        d2 = self.decoder2(d3)
        d2 = self.WF2(d2, e1)

        d1 = self.decoder1(d2)
        out = self.final(d1)

        result['out'] = out
        return result
