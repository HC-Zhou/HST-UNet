# -*- coding:utf-8  -*-
"""
Time: 2022/8/10 14:53
Author: Yimohanser
Software: PyCharm
"""
import glob
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm


def mean_std(img, w=256, h=256):
    count = 0
    R = 0
    R_channel_square = 0
    G = 0
    G_channel_square = 0
    B = 0
    B_channel_square = 0
    t = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])
    for i, m in enumerate(tqdm(img)):
        m = Image.open(m)
        m = t(m)
        count += w * h
        R += torch.sum(m[0, :, :])
        R_channel_square += torch.sum(torch.pow(m[0, :, :], 2.0))
        G += torch.sum(m[1, :, :])
        G_channel_square += torch.sum(torch.pow(m[1, :, :], 2.0))
        B += torch.sum(m[2, :, :])
        B_channel_square += torch.sum(torch.pow(m[2, :, :], 2.0))

    R_mean = R / count
    G_mean = G / count
    B_mean = B / count

    R_std = np.sqrt(R_channel_square / count - R_mean * R_mean)
    G_std = np.sqrt(G_channel_square / count - G_mean * G_mean)
    B_std = np.sqrt(B_channel_square / count - B_mean * B_mean)

    return [round(R_mean.item(), 3), round(G_mean.item(), 3), round(B_mean.item(), 3)], \
           [round(R_std.item(), 3), round(G_std.item(), 3), round(B_std.item(), 3)]


def read_root(root):
    return glob.glob(os.path.join(root, '*.tif'))


if __name__ == '__main__':
    imgs = read_root(root='F:\data\MathorCup\ext\img')
    print(mean_std(imgs, w=512, h=512))
