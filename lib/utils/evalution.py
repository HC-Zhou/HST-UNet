# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 14:42
Author: Yimohanser
Software: PyCharm
"""
import numpy as np


class Evaluator(object):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        np.seterr(divide='ignore', invalid='ignore')

    def OA(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Precision(self):
        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(1)
        return precision

    def Recall(self):
        recall = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(0))
        return recall

    def F1Score(self):
        precision = self.Precision()
        recall = self.Recall()
        return (2 * precision * recall) / (precision + recall)

    def IoU(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return IoU

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, pre_image, gt_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def getScore(self):
        pixel_acc = self.OA()
        precision = np.nanmean(self.Precision().tolist()[:-1])
        recall = np.nanmean(self.Recall().tolist()[:-1])
        f1_score = np.nanmean(self.F1Score().tolist()[:-1])
        mIOU = np.nanmean(self.Mean_Intersection_over_Union().tolist()[:-1])
        return [pixel_acc, precision, recall, f1_score, mIOU]

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
