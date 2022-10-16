# -*- coding:utf-8  -*-
"""
Time: 2022/9/13 8:21
Author: Yimohanser
Software: PyCharm
"""
import torch
from torch import nn
from collections import OrderedDict
from lib.sseg.HST.SSA import DropPath, shunted_s


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


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.1,
                 act_layer=nn.GELU
                 ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.act1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=2,
                                          output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.deconv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x


class HST_UNet(nn.Module):
    def __init__(self, img_size=256, pretrained='', num_classes=1):
        super(HST_UNet, self).__init__()
        self.encoder = shunted_s(img_size=img_size, pretrained=pretrained)

        self.LWF4 = LWF(256)
        self.decoder4 = UpSampleBlock(in_channels=512, out_channels=256)
        self.attn4 = nn.ModuleList([Block(256) for _ in range(4)])

        self.LWF3 = LWF(128)
        self.decoder3 = UpSampleBlock(in_channels=256, out_channels=128)
        self.attn3 = nn.ModuleList([Block(128) for _ in range(2)])

        self.LWF2 = LWF(64)
        self.decoder2 = UpSampleBlock(in_channels=128, out_channels=64)
        self.attn2 = nn.ModuleList([Block(64) for _ in range(2)])

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
        d4 = self.LWF4(d4, e3)
        for blk in self.attn4:
            d4 = blk(d4)

        d3 = self.decoder3(d4)
        d3 = self.LWF3(d3, e2)
        for blk in self.attn3:
            d3 = blk(d3)

        d2 = self.decoder2(d3)
        d2 = self.LWF2(d2, e1)
        for blk in self.attn2:
            d2 = blk(d2)

        d1 = self.decoder1(d2)
        out = self.final(d1)
        result['out'] = out
        return result


if __name__ == "__main__":
    from torchsummary import summary

    model = HST_UNet(img_size=256)
    summary(model, input_size=(3, 256, 256), device='cpu')
