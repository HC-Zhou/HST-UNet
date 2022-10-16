# -*- coding:utf-8  -*-
"""
Time: 2022/10/16 10:27
Author: Yimohanser
Software: PyCharm
"""
import torch
chkpt = torch.load("D:\ML\HST-UNet\weight\Potsdam.pth", map_location='cpu')
torch.save({"model": chkpt["model"]}, "Potsdam.pth")