# -*- coding:utf-8  -*-
"""
Time: 2022/9/17 13:40
Author: Yimohanser
Software: PyCharm
"""
# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 9:45
Author: Yimohanser
Software: PyCharm
"""
import torch
from PIL import Image
from torchvision import transforms
from .base_dataset import MultiClassBaseDataset
from ..utils.ext_transforms import ExtendTransforms


class Potsdam(MultiClassBaseDataset):
    def __init__(self,
                 root='',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 img_size=512,
                 mode='train'):
        colormap = [
            [255, 255, 255],  # Surfaces
            [0, 0, 255],  # Building
            [0, 255, 255],  # low Veg
            [0, 255, 0],  # Tree
            [255, 255, 0],  # Car
            [255, 0, 0]  # background
        ]
        super(Potsdam, self).__init__(root=root, colormap=colormap, mean=mean,
                                      std=std, img_size=img_size, mode=mode)
        self.x_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.y_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # Read Data
        item = self.files[self.mode][idx]
        image_path = item['img']
        mask_path = item['mask']

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Data Augmentation
        if self.mode == 'train':
            image, mask = ExtendTransforms.flip(image, mask)
            image, mask = ExtendTransforms.randomResizeCrop(image, mask)
            image, _ = ExtendTransforms.adjust_hue(image, mask)
            image, _ = ExtendTransforms.adjustBrightness(image, mask)
            image, _ = ExtendTransforms.adjustContrast(image, mask)
            image, _ = ExtendTransforms.adjustSaturation(image, mask)

        # Resize image
        img = self.x_transform(image)
        mask = self._image2label(self.y_transform(mask))
        mask = torch.from_numpy(mask)
        ori = self.img_transform(image)

        return img, mask, ori
