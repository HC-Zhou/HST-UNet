# -*- coding:utf-8  -*-
"""
Time: 2022/8/3 19:42
Author: Yimohanser
Software: PyCharm
"""
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from lib.optim import Ranger
from lib.datasets.Potsdam import Potsdam
from lib.utils.utils import create_logging
from lib.core.function import MultiClass
from lib.sseg import SSegmentationSet
from lib.criterion import CriterionSet
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')
    # Model parameters
    parser.add_argument('--model', default='hst_unet', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--img_size', default=256, type=int)

    # Dataset parameters
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--root', default=r'G:\Vaihingen\Vaihingen', type=str,
                        help='dataset path')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--workers', default=2, type=int)

    # Cudnn parameters
    parser.add_argument('--BENCHMARK', type=bool, default=True)
    parser.add_argument('--DETERMINISTIC', type=bool, default=False)
    parser.add_argument('--ENABLED', type=bool, default=True)

    # Logging parameters
    parser.add_argument('--log_path', default='./saveModels/logging/',
                        help='path where to tensorboard log')
    parser.add_argument('--weight', default='./saveModels/logging/best.pth',
                        help='model weight path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('chkpt path : ', args.log_path)

    writer = SummaryWriter(args.log_path)
    logger = create_logging(args, args.model, 'test')

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # cudnn related setting
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED

    device = torch.device(args.device)
    logger.info('Use device:{}'.format(args.device))

    # build model
    model = SSegmentationSet(model=args.model.lower(),
                             num_classes=args.num_classes,
                             pretrained='',
                             img_size=args.img_size)
    print(model)
    test_dataset = Potsdam(root=args.root,
                           mean=[0.337, 0.361, 0.334], std=[0.141, 0.138, 0.143],
                           img_size=args.img_size, mode='test')
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    classes = ['Surfaces', 'Building', 'low Veg', 'Tree', 'Car', 'BackGround']

    # criterion
    criterion = CriterionSet(loss='cross_entropy')

    # optimizer
    model = model.to(device)

    checkpoint = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    MultiClass.test(model=model, dataloader=testloader, criterion=criterion,
                    log_path=args.log_path,
                    class_name=classes, writer=writer, logging=logger, epoch=0,
                    num_classes=args.num_classes, device=device)


if __name__ == '__main__':
    main()
