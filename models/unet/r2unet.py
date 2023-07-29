# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:16:21 2023

@author: Gavin
"""

from torch import nn
from .._core import RREL2D, RRDL2D

class R2UNet(nn.Module):
    
    def __init__(self, channels, img_channels=1):
        super().__init__()
        
        self.rrel1 = RREL2D(img_channels, channels, 1)
        self.rrel2 = RREL2D(channels, channels * 2, 2)
        self.rrel3 = RREL2D(channels * 2, channels * 4, 2)
        self.rrel4 = RREL2D(channels * 4, channels * 8, 2)
        self.rrel5 = RREL2D(channels * 8, channels * 16, 2)
        
        self.rrdl6 = RRDL2D(channels * 16, channels * 8, 2)
        self.rrdl7 = RRDL2D(channels * 8, channels * 4, 2)
        self.rrdl8 = RRDL2D(channels * 4, channels * 2, 2)
        self.rrdl9 = RRDL2D(channels * 2, channels, 2)
        
        self.cn10 = nn.Conv3d(
            channels,
            1,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            stride=(1, 1, 1)
        )
        self.a10 = nn.Sigmoid()
        
    
    
    def forward(self, x):
        out1 = self.rrel1(x)
        out2 = self.rrel2(out1)
        out3 = self.rrel3(out2)
        out4 = self.rrel4(out3)
        out5 = self.rrel5(out4)
        
        out6 = self.rrdl6(out5, out4)
        out7 = self.rrdl7(out6, out3)
        out8 = self.rrdl8(out7, out2)
        out9 = self.rrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10
    


class R2UNet360(R2UNet):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)
        
        self.rrel5 = RREL2D(channels * 8, channels * 16, 3)
        self.rrdl6 = RRDL2D(channels * 16, channels * 8, 3)
        
        