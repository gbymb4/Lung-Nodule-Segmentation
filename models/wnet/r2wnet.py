# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:16:30 2023

@author: Gavin
"""

from torch import nn
from .._core import DRREL, DRRDL

class R2WNet(nn.Module):
    
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
        
        self.drrel14 = DRREL(channels, channels * 2, 2)
        self.drrel15 = DRREL(channels * 2, channels * 4, 2)
        self.drrel16 = DRREL(channels * 4, channels * 8, 2)
        self.drrel17 = DRREL(channels * 8, channels * 16, 3)
        self.drrel18 = DRREL(channels * 16, channels * 16, 3)
        self.drrel19 = DRREL(channels * 16, channels * 16, 5)
        
        self.drrdl20 = DRRDL(channels * 16, channels * 16, 5)
        self.drrdl21 = DRRDL(channels * 16, channels * 16, 3)
        self.drrdl22 = DRRDL(channels * 16, channels * 8, 3)
        self.drrdl23 = DRRDL(channels * 8, channels * 4, 2)
        self.drrdl24 = DRRDL(channels * 4, channels * 2, 2)
        self.drrdl25 = DRRDL(channels * 2, channels, 2)
        
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
        
        # enc-dec 2
        
        out14 = self.drrel14(out13)
        out15 = self.drrel15(out14)
        out16 = self.drrel16(out15)
        out17 = self.drrel17(out16)
        out18 = self.drrel18(out17)
        out19 = self.drrel19(out18)
        
        out20 = self.drrdl20(out19, out18)
        out21 = self.drrdl21(out20, out17)
        out22 = self.drrdl22(out21, out16)
        out23 = self.drrdl23(out22, out15)
        out24 = self.drrdl24(out23, out14)
        out25 = self.drrdl25(out24, out13)
        
        # element-wise sum and output convolution
        
        out26 = self.a26(self.cn26(out13 + out25))
        
        return out26