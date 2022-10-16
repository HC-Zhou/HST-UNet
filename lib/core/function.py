# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 14:04
Author: Yimohanser
Software: PyCharm
"""
import math
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from lib.utils.utils import MetricLogger, writeData, plot_img_mask, SmoothedValue, plot_img_gray
from lib.utils.evalution import Evaluator


class MultiClass:
    @staticmethod
    def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
              dataloader: DataLoader, epoch: int, num_classes, log_path, class_name,
              logging, writer, scaler, device):
        model.train(True)
        optimizer.zero_grad()

        logging.info('Training Epoch[{}]:'.format(epoch))
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        Eval = Evaluator(num_classes)

        for i_iter, (img, mask, ori) in enumerate(
                metric_logger.log_every(
                    dataloader, print_freq=(len(dataloader) // 100) if len(dataloader) >= 400 else 20,
                    header=header
                )):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                out = model(img)
                loss = criterion(out, mask)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])

            pred_mask = out['out'].argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            mask_list = mask.detach().cpu().numpy().astype(np.uint8)
            Eval.add_batch(pred_mask, mask_list)

        measure = Eval.getScore()
        F1_score = Eval.F1Score().tolist()[:-1]
        IoU = Eval.IoU().tolist()[:-1]

        plot_img_mask(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='train')
        writeData(writer=writer, logger=logging, lr=optimizer.state_dict()['param_groups'][0]['lr'],
                  measure=measure, F1_class=F1_score, loss=metric_logger.meters["loss"].global_avg,
                  epoch=epoch, mode='train', classes=class_name, IoU=IoU)
        torch.cuda.empty_cache()

    @staticmethod
    @torch.no_grad()
    def val(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
            num_classes, class_name, writer, logging, epoch, log_path, device):
        model.eval()

        logging.info('Valid Epoch[{}]:'.format(epoch))
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        Eval = Evaluator(num_classes)

        for i_iter, (img, mask, ori) in enumerate(
                metric_logger.log_every(
                    dataloader, print_freq=(len(dataloader) // 50) if len(dataloader) >= 200 else 20,
                    header=header
                )):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            out = model(img)
            loss = criterion(out, mask)

            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            pred_mask = out['out'].argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            mask_list = mask.detach().cpu().numpy().astype(np.uint8)
            Eval.add_batch(pred_mask, mask_list)

        measure = Eval.getScore()
        F1_score = Eval.F1Score().tolist()[:-1]
        IoU = Eval.IoU().tolist()[:-1]

        plot_img_mask(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='val')
        writeData(writer=writer, logger=logging, measure=measure, F1_class=F1_score,
                  loss=metric_logger.meters["loss"].global_avg, epoch=epoch, mode='val', classes=class_name,
                  IoU=IoU)
        torch.cuda.empty_cache()

    @staticmethod
    @torch.no_grad()
    def test(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             num_classes, class_name, writer, logging, log_path, epoch, device):
        model.eval()

        logging.info('Test Epoch[{}]:'.format(epoch))
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        Eval = Evaluator(num_classes)

        for i_iter, (img, mask, ori) in enumerate(
                metric_logger.log_every(
                    dataloader, print_freq=(len(dataloader) // 50) if len(dataloader) >= 200 else 20,
                    header=header
                )):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            out = model(img)
            loss = criterion(out, mask)

            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            pred_mask = out['out'].argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            mask_list = mask.detach().cpu().numpy().astype(np.uint8)
            Eval.add_batch(pred_mask, mask_list)

        measure = Eval.getScore()
        F1_score = Eval.F1Score().tolist()[:-1]
        IoU = Eval.IoU().tolist()[:-1]
        plot_img_mask(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='test')
        writeData(writer=writer, logger=logging, measure=measure, F1_class=F1_score,
                  loss=metric_logger.meters["loss"].global_avg, epoch=epoch, mode='test', classes=class_name,
                  IoU=IoU)
        torch.cuda.empty_cache()
        return IoU, np.nanmean(IoU), np.nanmean(F1_score)


class SingleClass:
    @staticmethod
    def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, dataloader: DataLoader,
              epoch, class_name, log_path, logging, writer, scaler, device):
        model.train(True)
        optimizer.zero_grad()

        logging.info('Training: Epoch[{}]'.format(epoch))
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        Eval = Evaluator(2)
        for i_iter, (img, mask, ori) in enumerate(
                metric_logger.log_every(
                    dataloader, print_freq=(len(dataloader) // 100) if len(dataloader) >= 200 else 20,
                    header=header
                )):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                out = model(img)
                loss = criterion(out, mask)

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])

            pred_mask = out['out'].sigmoid().squeeze(1).detach().cpu().numpy()
            mask_list = mask.squeeze(1).detach().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            mask_list = mask_list.astype(np.uint8)

            Eval.add_batch(pred_mask, mask_list)

        measure = Eval.getScore()
        F1_score = Eval.F1Score().tolist()
        plot_img_gray(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='train')
        writeData(writer=writer, logger=logging, lr=optimizer.state_dict()['param_groups'][0]['lr'],
                  measure=measure, F1_class=F1_score, loss=metric_logger.meters["loss"].global_avg,
                  epoch=epoch, mode='train', classes=class_name)
        torch.cuda.empty_cache()

    @staticmethod
    @torch.no_grad()
    def val(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
            class_name, writer, logging, epoch, log_path, device):
        model.eval()

        logging.info('Valid Epoch[{}]:'.format(epoch))
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        Eval = Evaluator(2)

        for i_iter, (img, mask, ori) in enumerate(
                metric_logger.log_every(
                    dataloader, print_freq=(len(dataloader) // 50) if len(dataloader) >= 100 else 20,
                    header=header
                )):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            out = model(img)
            loss = criterion(out, mask)

            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            pred_mask = out['out'].sigmoid().squeeze(1).detach().cpu().numpy()
            mask_list = mask.squeeze(1).detach().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            mask_list = mask_list.astype(np.uint8)
            Eval.add_batch(pred_mask, mask_list)

        measure = Eval.getScore()
        F1_score = Eval.F1Score().tolist()
        plot_img_gray(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='val')
        writeData(writer=writer, logger=logging, measure=measure, F1_class=F1_score,
                  loss=metric_logger.meters["loss"].global_avg, epoch=epoch, mode='val', classes=class_name)
        torch.cuda.empty_cache()

    @staticmethod
    @torch.no_grad()
    def test(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             class_name, writer, logging, epoch, log_path, device):
        model.eval()

        logging.info('Valid Epoch[{}]:'.format(epoch))
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        Eval = Evaluator(2)

        for i_iter, (img, mask, ori) in enumerate(
                metric_logger.log_every(
                    dataloader, print_freq=(len(dataloader) // 50) if len(dataloader) >= 100 else 20,
                    header=header
                )):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            out = model(img)
            loss = criterion(out, mask)

            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            pred_mask = out['out'].sigmoid().squeeze(1).detach().cpu().numpy()
            mask_list = mask.squeeze(1).detach().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            mask_list = mask_list.astype(np.uint8)
            Eval.add_batch(pred_mask, mask_list)

        measure = Eval.getScore()
        F1_score = Eval.F1Score().tolist()
        plot_img_gray(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='test')
        writeData(writer=writer, logger=logging, measure=measure, F1_class=F1_score,
                  loss=metric_logger.meters["loss"].global_avg, epoch=epoch, mode='test', classes=class_name)
        torch.cuda.empty_cache()
