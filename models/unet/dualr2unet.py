# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:16:48 2023

@author: Gavin
"""

from torch import nn
from .._core import DRREL, DRRDL

class DualR2UNet(nn.Module):
    
    def __init__(self, channels, img_channels=1):
        super().__init__()
        
        self.drrel1 = DRREL(img_channels, channels, 1)
        self.drrel2 = DRREL(channels, channels * 2, 2)
        self.drrel3 = DRREL(channels * 2, channels * 4, 2)
        self.drrel4 = DRREL(channels * 4, channels * 8, 2)
        self.drrel5 = DRREL(channels * 8, channels * 16, 3)
        self.drrel6 = DRREL(channels * 16, channels * 16, 3)
        self.drrel7 = DRREL(channels * 16, channels * 16, 5)
        
        self.drrdl8 = DRRDL(channels * 16, channels * 16, 5)
        self.drrdl9 = DRRDL(channels * 16, channels * 16, 3)
        self.drrdl10 = DRRDL(channels * 16, channels * 8, 3)
        self.drrdl11 = DRRDL(channels * 8, channels * 4, 2)
        self.drrdl12 = DRRDL(channels * 4, channels * 2, 2)
        self.drrdl13 = DRRDL(channels * 2, channels, 2)
        
        self.cn14 = nn.Conv3d(
            channels,
            1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        self.a14 = nn.Sigmoid()
        
    
    
    def forward(self, x):
        out1 = self.drrel1(x)
        out2 = self.drrel2(out1)
        out3 = self.drrel3(out2)
        out4 = self.drrel4(out3)
        out5 = self.drrel5(out4)
        out6 = self.drrel6(out5)
        out7 = self.drrel7(out6)
        
        out8 = self.drrdl8(out7, out6)
        out9 = self.drrdl9(out8, out5)
        out10 = self.drrdl10(out9, out4)
        out11 = self.drrdl11(out10, out3)
        out12 = self.drrdl12(out11, out2)
        out13 = self.drrdl13(out12, out1)
        
        out14 = self.a14(self.cn14(out13))
        
        return out14
        