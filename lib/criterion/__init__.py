# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 10:39
Author: Yimohanser
Software: PyCharm
"""
import torch
from collections import OrderedDict
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch.nn import functional as F
from lib.criterion.bce_dice import dice_bce_loss

def CriterionSet(loss='cross_entropy'):
    if loss == 'cross_entropy':
        return CrossEntropyLoss()
    elif loss == 'bce_dice_loss':
        return BCE_Dice_Loss()
    elif loss == 'bce_logits_loss':
        return BCEWithLogitsLoss()


class BCEWithLogitsLoss:
    def __call__(self, inputs, target):
        losses = {}
        for name, x in inputs.items():
            xh, xw = x.size(2), x.size(3)
            h, w = target.size(2), target.size(3)

            if xh != h or xw != w:
                x = F.interpolate(
                    input=x, size=(h, w),
                    mode='bilinear', align_corners=True
                )

            losses[name] = binary_cross_entropy_with_logits(x, target)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']


class CrossEntropyLoss:
    def __call__(self, inputs, target):
        losses = {}
        for name, x in inputs.items():
            xh, xw = x.size(2), x.size(3)
            h, w = target.size(1), target.size(2)

            if xh != h or xw != w:
                x = F.interpolate(
                    input=x, size=(h, w),
                    mode='bilinear', align_corners=True
                )

            losses[name] = cross_entropy(x, target)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']


class BCE_Dice_Loss:
    def __init__(self):
        self.loss_fn = dice_bce_loss()

    def __call__(self, inputs, target):
        losses = {}
        for name, x in inputs.items():
            xh, xw = x.size(2), x.size(3)
            h, w = target.size(2), target.size(3)

            if xh != h or xw != w:
                x = F.interpolate(
                    input=x, size=(h, w),
                    mode='bilinear', align_corners=True
                )

            losses[name] = self.loss_fn(x, target)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']


if __name__ == '__main__':
    torch.manual_seed(0)
    predict = OrderedDict()
    predict['out'] = torch.randn((1, 3, 3, 3), dtype=torch.float32)
    mask = torch.zeros((1, 3, 3), dtype=torch.long)
    loss = CrossEntropyLoss()
    print(loss(predict, mask))
