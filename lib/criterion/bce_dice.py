# -*- coding:utf-8  -*-
"""
Time: 2022/8/6 17:03
Author: Yimohanser
Software: PyCharm
"""
import torch.nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(dice_bce_loss, self).__init__()
        self.smooth = smooth
        self.dims = dims
        self.bce_loss = nn.BCEWithLogitsLoss()

    def SoftDiceLoss(self, y_pred, y_true):
        tp = (y_pred * y_true).sum(self.dims)
        fp = (y_pred * (1 - y_true)).sum(self.dims)
        fn = ((1 - y_pred) * y_true).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc

    def __call__(self, y_pred, y_true):
        bce = self.bce_loss(y_pred, y_true)
        dice = self.SoftDiceLoss(y_pred.sigmoid(), y_true)
        return 0.8 * bce + 0.2 * dice
