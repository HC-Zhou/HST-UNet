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
from lib.datasets.vaihingen import Vaihingen
from lib.utils.utils import create_logging
from lib.core.function import MultiClass
from lib.sseg import SSegmentationSet
from lib.criterion import CriterionSet
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    # Model parameters
    parser.add_argument('--model', default='hst_unet', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--pretrained', default='', type=str)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight decay (default: 0.001)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--root', default=r'G:\Vaihingen\Vaihingen', type=str,
                        help='dataset path')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')

    # Cudnn parameters
    parser.add_argument('--BENCHMARK', type=bool, default=True)
    parser.add_argument('--DETERMINISTIC', type=bool, default=False)
    parser.add_argument('--ENABLED', type=bool, default=True)

    # Logging parameters
    parser.add_argument('--log_path', default='./saveModels/logging/',
                        help='path where to tensorboard log')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.resume:
        count = 0
        while os.path.exists(
                args.log_path + args.model + "_Vaihingen_ver_" + str(count)):
            count += 1
        args.log_path = args.log_path + args.model + "_Vaihingen_ver_" + str(count)
        os.mkdir(args.log_path)
    print('chkpt path : ', args.log_path)

    writer = SummaryWriter(args.log_path)
    logger = create_logging(args, args.model, 'train')

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
                             pretrained=args.pretrained,
                             img_size=args.img_size)
    print(model)

    # prepare data
    train_dataset = Vaihingen(root=args.root,
                              mean=[0.469, 0.321, 0.318], std=[0.22, 0.16, 0.153],
                              img_size=args.img_size, mode='train')
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    test_dataset = Vaihingen(root=args.root,
                             mean=[0.469, 0.321, 0.318], std=[0.22, 0.16, 0.153],
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
    optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / (x + 1))

    # Evaluation
    Best_MIoU = 0.
    Best_F1 = 0.

    if args.resume:
        checkpoint = torch.load(args.log_path + '/checkpoint.pth', map_location='cpu')
        epoch_start = checkpoint['epoch']
        Best_MIoU = checkpoint['MIoU']
        Best_F1 = checkpoint['F1']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('------------RESUME TRAINING------------')
    else:
        epoch_start = 0

    # amp training
    scaler = GradScaler()
    for epoch in range(epoch_start, args.epochs):
        print('\nEpoch:{}/{}:'.format(epoch, args.epochs))
        MultiClass.train(model=model, optimizer=optimizer, criterion=criterion,
                         dataloader=trainloader, epoch=epoch, logging=logger, num_classes=args.num_classes,
                         writer=writer, scaler=scaler, device=device, log_path=args.log_path, class_name=classes)
        scheduler.step(epoch=epoch + 1)

        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'MIoU': Best_MIoU,
            'F1': Best_F1
        }, os.path.join(args.log_path, 'checkpoint.pth'))

        IoU, IoU_Score, F1_score = MultiClass.test(model=model, dataloader=testloader, criterion=criterion, log_path=args.log_path,
                        class_name=classes, writer=writer, logging=logger, epoch=epoch,
                        num_classes=args.num_classes, device=device)

        if IoU_Score > Best_MIoU and F1_score > Best_F1:
            Best_MIoU = IoU_Score
            Best_F1 = F1_score
            torch.save({
                'model': model.state_dict(),
                'MIoU': Best_MIoU,
                'F1': Best_F1
            }, os.path.join(args.log_path, f"Best{epoch}.pth"))


if __name__ == '__main__':
    main()
