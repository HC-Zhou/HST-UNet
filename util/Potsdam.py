# -*- coding:utf-8  -*-
"""
Time: 2022/9/17 10:42
Author: Yimohanser
Software: PyCharm
"""
import glob
import os
import torch
import tifffile
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

ColorMap = [
    [255, 255, 255],  # Surfaces
    [0, 0, 255],  # Building
    [0, 255, 255],  # low Veg
    [0, 255, 0],  # Tree
    [255, 255, 0],  # Car
    [255, 0, 0],  # background
]

crop_size = 256
root = 'G:\Potsdam'
test = ['2_13', '2_14', '3_13', '3_14', '4_13', '4_15',
        '5_13', '5_14', '5_15', '6_14', '6_15', '7_13']

ColorMap2lbl = np.zeros(256 ** 3, dtype=np.int64)
for idx, colormap in enumerate(ColorMap):
    ColorMap2lbl[((colormap[0] * 256 + colormap[1]) * 256) + colormap[2]] = idx

img_files = os.listdir(os.path.join(root, '2_Ortho_RGB'))

for idx, name in enumerate(tqdm(img_files)):
    # 读入图片
    img = tifffile.imread((
        os.path.join(root, '2_Ortho_RGB', name)))
    label = tifffile.imread(
        os.path.join(root, '5_Labels_all',
                     name.replace('RGB', 'label')))

    h, w = img.shape[:2]

    ID = name.split('_')[2] + '_' + name.split('_')[3]

    if ID in test:
        stride = crop_size
    else:
        stride = crop_size // 2

    # 枚举切割位置(以为滑动窗口步距)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            up, down = i, i + crop_size
            left, right = j, j + crop_size

            # 防止越界
            if down > h:
                down, up = h, h - crop_size

            if right > w:
                right, left = w, w - crop_size

            # 切割图像
            crop_img = img[up: down, left: right, :]
            crop_mask = label[up: down, left: right, :]

            image = Image.fromarray(np.uint8(crop_img))
            crop_mask = Image.fromarray(np.uint8(crop_mask))

            if ID in test:
                image.save(root + f'/test/img/{i}_{j}_{ID}.tif')
                crop_mask.save(root + f'/test/mask/{i}_{j}_{ID}.tif')
            else:
                image.save(root + f'/train/img/{i}_{j}_{ID}.tif')
                crop_mask.save(root + f'/train/mask/{i}_{j}_{ID}.tif')


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


train_files = os.listdir(r'G:\Potsdam\train\img')
test_files = os.listdir(r'G:\Potsdam\test\img')

train_txt = open(os.path.join(root, 'train', 'train.txt'), 'w')
for i, file in enumerate(tqdm(train_files)):
    train_txt.write(file + '\n')

test_txt = open(os.path.join(root, 'test', 'test.txt'), 'w')
for i, file in enumerate(tqdm(test_files)):
    test_txt.write(file + '\n')

print(mean_std(read_root(r'G:\Potsdam\train\img'), h=256, w=256))
