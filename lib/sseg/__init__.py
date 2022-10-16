# -*- coding:utf-8  -*-
"""
Time: 2022/8/3 19:34
Author: Yimohanser
Software: PyCharm
"""
from .HST import *

def SSegmentationSet(model: str, num_classes=5, pretrained='', img_size=256):
    if model == 'baseline':
        return HST_UNet_Baseline(img_size=img_size, pretrained=pretrained, num_classes=num_classes)
    elif model == 'hst_unet_wf':
        return HST_UNet_LWF(img_size=img_size, pretrained=pretrained, num_classes=num_classes)
    elif model == 'hst_unet_mscan':
        return HST_UNet_MSCAN(img_size=img_size, pretrained=pretrained, num_classes=num_classes)
    elif model == 'hst_unet':
        return HST_UNet(img_size=img_size, pretrained=pretrained, num_classes=num_classes)