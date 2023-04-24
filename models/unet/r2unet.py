# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:16:21 2023

@author: Gavin
"""

import torch

from torch import nn
from .._core import R2DBlock, RREL2D, RRDL2D

class R2UNet(nn.Module):
    
    def __init__(self, channels, img_channels=1):
        super().__init__()
        
        self.rrel1 = RREL2D(img_channels, channels, 1)
        self.rrel2 = RREL2D(channels, channels * 2, 2)
        self.rrel3 = RREL2D(channels * 2, channels * 4, 2)
        self.rrel4 = RREL2D(channels * 4, channels * 8, 2)
        self.rrel5 = RREL2D(channels * 8, channels * 16, 3)
        self.rrel6 = RREL2D(channels * 16, channels * 16, 3)
        self.rrel7 = RREL2D(channels * 16, channels * 16, 5)
        
        self.rrdl8 = RRDL2D(channels * 16, channels * 16, 5)
        self.rrdl9 = RRDL2D(channels * 16, channels * 16, 3)
        self.rrdl10 = RRDL2D(channels * 16, channels * 8, 3)
        self.rrdl11 = RRDL2D(channels * 8, channels * 4, 2)
        self.rrdl12 = RRDL2D(channels * 4, channels * 2, 2)
        self.rrdl13 = RRDL2D(channels * 2, channels, 2)
        
        self.cn14 = nn.Conv3d(
            channels,
            1,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        self.a14 = nn.Sigmoid()
        
    
    
    def forward(self, x):
        out1 = self.rrel1(x)
        out2 = self.rrel2(out1)
        out3 = self.rrel3(out2)
        out4 = self.rrel4(out3)
        out5 = self.rrel5(out4)
        out6 = self.rrel6(out5)
        out7 = self.rrel7(out6)
        
        out8 = self.rrdl8(out7, out6)
        out9 = self.rrdl9(out8, out5)
        out10 = self.rrdl10(out9, out4)
        out11 = self.rrdl11(out10, out3)
        out12 = self.rrdl12(out11, out2)
        out13 = self.rrdl13(out12, out1)
        
        out14 = self.a14(self.cn14(out13))
        
        return out14
        