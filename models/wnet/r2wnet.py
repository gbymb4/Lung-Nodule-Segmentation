# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:16:30 2023

@author: Gavin
"""

from torch import nn
from .._core import RREL2D, RRDL2D

class R2WNet(nn.Module):
    
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
        
        self.rrel14 = RREL2D(channels, channels * 2, 2)
        self.rrel15 = RREL2D(channels * 2, channels * 4, 2)
        self.rrel16 = RREL2D(channels * 4, channels * 8, 2)
        self.rrel17 = RREL2D(channels * 8, channels * 16, 3)
        self.rrel18 = RREL2D(channels * 16, channels * 16, 3)
        self.rrel19 = RREL2D(channels * 16, channels * 16, 5)
        
        self.rrdl20 = RRDL2D(channels * 16, channels * 16, 5)
        self.rrdl21 = RRDL2D(channels * 16, channels * 16, 3)
        self.rrdl22 = RRDL2D(channels * 16, channels * 8, 3)
        self.rrdl23 = RRDL2D(channels * 8, channels * 4, 2)
        self.rrdl24 = RRDL2D(channels * 4, channels * 2, 2)
        self.rrdl25 = RRDL2D(channels * 2, channels, 2)
        
        self.cn26 = nn.Conv3d(
            channels,
            1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        self.a26 = nn.Sigmoid()
        
    
    
    def forward(self, x):
        # enc-dec 1
        
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
        
        # enc-dec 2
        
        out14 = self.rrel14(out13)
        out15 = self.rrel15(out14)
        out16 = self.rrel16(out15)
        out17 = self.rrel17(out16)
        out18 = self.rrel18(out17)
        out19 = self.rrel19(out18)
        
        out20 = self.rrdl20(out19, out18)
        out21 = self.rrdl21(out20, out17)
        out22 = self.rrdl22(out21, out16)
        out23 = self.rrdl23(out22, out15)
        out24 = self.rrdl24(out23, out14)
        out25 = self.rrdl25(out24, out13)
        
        # element-wise sum and output convolution
        
        out26 = self.a26(self.cn26(out13 + out25))
        
        return out26