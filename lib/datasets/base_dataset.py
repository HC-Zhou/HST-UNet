# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 9:17
Author: Yimohanser
Software: PyCharm
"""
import collections
import numpy as np
from abc import abstractmethod
from torch.utils import data


class MultiClassBaseDataset(data.Dataset):
    def __init__(self,
                 root,
                 colormap=[[255, 255, 255]],
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 img_size=512,
                 mode='train'):
        super(MultiClassBaseDataset, self).__init__()
        self.root = root
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.mode = mode
        self.colormap = colormap
        self.colormap2lbl = np.zeros(256 ** 3)

        for idx, colormap in enumerate(self.colormap):
            self.colormap2lbl[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = idx

        self.files = collections.defaultdict(list)
        for split_file in ['train', 'test']:
            ImageSet_file = self.root + f'/{split_file}/{split_file}.txt'
            for img_name in open(ImageSet_file):
                img_name = img_name.strip()
                img_file = self.root + f'/{split_file}/img/{img_name}'
                mask_file = self.root + f'/{split_file}/mask/{img_name}'
                self.files[split_file].append({
                    'img': img_file,
                    'mask': mask_file
                })

    def _image2label(self, mask):
        mask = np.array(mask, dtype=np.int64)
        idx = (mask[:, :, 0] * 256 + mask[:, :, 1]) * 256 + mask[:, :, 2]
        return np.array(self.colormap2lbl[idx], dtype=np.int64)

    @abstractmethod
    def __getitem__(self, idx):
        return

    def __len__(self):
        return len(self.files[self.mode])


class SingleClassBaseDataset(data.Dataset):
    def __init__(self, root,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 img_size=512,
                 mode='train'):
        super(SingleClassBaseDataset, self).__init__()
        self.mode = mode
        self.root = root
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.files = collections.defaultdict(list)
        for split_file in ['train', 'val']:
            imgsets_file = self.root + f'/{split_file}.txt'
            for img_name in open(imgsets_file):
                img_name = img_name.strip()
                img_file = self.root + f'/img/{img_name}'
                lbl_file = self.root + f'/label/{img_name}'
                self.files[split_file].append({
                    'img': img_file,
                    'mask': lbl_file,
                })

    @abstractmethod
    def __getitem__(self, idx):
        return

    def __len__(self):
        return len(self.files[self.mode])