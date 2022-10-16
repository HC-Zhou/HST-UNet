# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 9:54
Author: Yimohanser
Software: PyCharm
"""
import random
import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image
from torchvision import transforms


class ExtendTransforms:
    def __init__(self):
        super().__init__()

    @staticmethod
    def rotate(image, mask, angle=None, p=0.3, fill=[255, 0, 0]):
        if random.random() < p:
            if angle is None:
                # -180~180随机选一个角度旋转
                angle = transforms.RandomRotation.get_params([-180, 180])
            if isinstance(angle, list):
                angle = random.choice(angle)
            image = image.rotate(angle)
            mask = mask.rotate(angle, fillcolor=fill)

        return image, mask

    @staticmethod
    def flip(image, mask, p=0.3):
        if random.random() < p:
            # 水平翻转
            if random.random() < 0.5:
                image = tf.hflip(image)
                mask = tf.hflip(mask)
            # 垂直翻转
            if random.random() > 0.5:
                image = tf.vflip(image)
                mask = tf.vflip(mask)
        return image, mask

    @staticmethod
    def randomResizeCrop(image, mask, scale=(0.3, 1.0), ratio=(0.5, 1), p=0.3):
        # scale表示随机crop出来的图片会在的0.3倍至1倍之间，ratio表示长宽比
        if random.random() < p:
            img = np.array(image)
            h_image, w_image = img.shape[0], img.shape[1]
            resize_size = h_image

            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
            image = tf.resized_crop(image, i, j, h, w, resize_size,
                                    interpolation=transforms.InterpolationMode.BILINEAR)
            mask = tf.resized_crop(mask, i, j, h, w, resize_size,
                                   interpolation=transforms.InterpolationMode.NEAREST)

        return image, mask

    @staticmethod
    def adjustContrast(image, mask, factor=0.15, p=0.3):
        # 调整图片对比度
        if random.random() < p:
            factor = random.uniform(1 - factor, 1 + factor)
            image = tf.adjust_contrast(image, factor)

        return image, mask

    @staticmethod
    def adjust_hue(image, mask, factor=0.15, p=0.3):
        # 调整图片色调
        if random.random() < p:
            factor = random.uniform(-factor, factor)
            image = tf.adjust_hue(image, factor)

        return image, mask

    @staticmethod
    def adjustBrightness(image, mask, factor=0.15, p=0.3):
        if random.random() < p:
            factor = random.uniform(1 - factor, 1 + factor)
            image = tf.adjust_brightness(image, factor)

        return image, mask

    @staticmethod
    def adjustSaturation(image, mask, factor=0.15, p=0.3):
        # 调整饱和度
        if random.random() < p:
            factor = random.uniform(1 - factor, 1 + factor)
            image = tf.adjust_saturation(image, factor)

        return image, mask

    @staticmethod
    def centerCrop(image, mask, size=None, p=0.3):
        # 中心裁剪
        if random.random() < p:
            real_size = image.size
            if size is None:
                size = image.size  # 若不设定size，则是原图。
            image = tf.center_crop(image, size)
            mask = tf.center_crop(mask, size)

        return image, mask

    @staticmethod
    def add_gaussian_noise(image, mask, noise_sigma=25, p=0.3):
        """高斯噪声"""
        if random.random() < p:
            temp_image = np.float64(np.copy(image))
            h, w, _ = temp_image.shape
            # 标准正态分布*noise_sigma
            noise = np.random.randn(h, w) * noise_sigma
            noisy_image = np.zeros(temp_image.shape, np.float64)
            if len(temp_image.shape) == 2:
                noisy_image = temp_image + noise
            else:
                noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
                noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
                noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
            noisy_image = Image.fromarray(np.uint8(noisy_image))
            image = noisy_image

        return image, mask
