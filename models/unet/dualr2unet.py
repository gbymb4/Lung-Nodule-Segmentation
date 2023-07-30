# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:16:48 2023

@author: Gavin
"""

from torch import nn
from .._core import DRREL, DRRDL, RREL3D, RRDL3D

class DualR2UNetEnc(nn.Module):
    
    def __init__(self, channels, img_channels=1):
        super().__init__()
        
        self.drrel1 = DRREL(img_channels, channels, 1)
        self.drrel2 = DRREL(channels, channels * 2, 2)
        self.drrel3 = DRREL(channels * 2, channels * 4, 2)
        self.drrel4 = DRREL(channels * 4, channels * 8, 2)
        self.drrel5 = DRREL(channels * 8, channels * 16, 2)
        
        self.rrdl6 = RRDL3D(channels * 16, channels * 8, 2)
        self.rrdl7 = RRDL3D(channels * 8, channels * 4, 2)
        self.rrdl8 = RRDL3D(channels * 4, channels * 2, 2)
        self.rrdl9 = RRDL3D(channels * 2, channels, 2)
        
        self.cn10 = nn.Conv3d(
            channels,
            1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        self.a10 = nn.Sigmoid()
        
    
    
    def forward(self, x):
        out1 = self.drrel1(x)
        out2 = self.drrel2(out1)
        out3 = self.drrel3(out2)
        out4 = self.drrel4(out3)
        out5 = self.drrel5(out4)
        
        out6 = self.rrdl6(out5, out4)
        out7 = self.rrdl7(out6, out3)
        out8 = self.rrdl8(out7, out2)
        out9 = self.rrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10
        
    
    
class DualR2UNet360Enc(DualR2UNetEnc):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)
        
        self.drrel5 = DRREL(channels * 8, channels * 16, 3)
        self.drrdl6 = RRDL3D(channels * 16, channels * 8, 3)
        
        
    
class DualR2UNetDec(nn.Module):
    
    def __init__(self, channels, img_channels=1):
        super().__init__()
        
        self.drrel1 = RREL3D(img_channels, channels, 1)
        self.drrel2 = RREL3D(channels, channels * 2, 2)
        self.drrel3 = RREL3D(channels * 2, channels * 4, 2)
        self.drrel4 = RREL3D(channels * 4, channels * 8, 2)
        self.drrel5 = RREL3D(channels * 8, channels * 16, 2)
        
        self.rrdl6 = DRRDL(channels * 16, channels * 8, 2)
        self.rrdl7 = DRRDL(channels * 8, channels * 4, 2)
        self.rrdl8 = DRRDL(channels * 4, channels * 2, 2)
        self.rrdl9 = DRRDL(channels * 2, channels, 2)
        
        self.cn10 = nn.Conv3d(
            channels,
            1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        self.a10 = nn.Sigmoid()
        
    
    
    def forward(self, x):
        out1 = self.rrel1(x)
        out2 = self.rrel2(out1)
        out3 = self.rrel3(out2)
        out4 = self.rrel4(out3)
        out5 = self.rrel5(out4)
        
        out6 = self.drrdl6(out5, out4)
        out7 = self.drrdl7(out6, out3)
        out8 = self.drrdl8(out7, out2)
        out9 = self.drrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10
        
    
    
class DualR2UNet360Dec(DualR2UNetDec):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)
        
        self.rrel5 = DRREL(channels * 8, channels * 16, 3)
        self.drrdl6 = RRDL3D(channels * 16, channels * 8, 3)