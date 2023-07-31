# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:21:05 2023

@author: Gavin
"""

from .dualr2unet import DualR2UNetEnc, DualR2UNet360Enc 
from .._core import AttentionBlock

class SADualR2UNet(DualR2UNetEnc):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)
        
        self.att2 = AttentionBlock(channels * 2)
        self.att3 = AttentionBlock(channels * 4)
        self.att4 = AttentionBlock(channels * 8)
        self.att5 = AttentionBlock(channels * 16)
        
    
    
    def forward(self, x):
        out1 = self.drrel1(x)
        
        out2 = self.drrel2(out1)
        out2 += self.att2(out2) * out2
        
        out3 = self.drrel3(out2)
        out3 += self.att3(out3) * out3
        
        out4 = self.drrel4(out3)
        out4 += self.att4(out4) * out4
        
        out5 = self.drrel5(out4)
        out5 += self.att5(out5) * out5
        
        out6 = self.rrdl6(out5, out4)
        out7 = self.rrdl7(out6, out3)
        out8 = self.rrdl8(out7, out2)
        out9 = self.rrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10
    
    
    
class SADualR2UNet360(DualR2UNet360Enc):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)
        
        self.att2 = AttentionBlock(channels)
        self.att3 = AttentionBlock(channels)
        self.att4 = AttentionBlock(channels)
        self.att5 = AttentionBlock(channels)
        
    
    
    def forward(self, x):
        out1 = self.drrel1(x)
        
        out2 = self.drrel2(out1)
        out2 += self.att2(out2) * out2
        
        out3 = self.drrel3(out2)
        out3 += self.att3(out3) * out3
        
        out4 = self.drrel4(out3)
        out4 += self.att4(out4) * out4
        
        out5 = self.drrel5(out4)
        out5 += self.att5(out5) * out5
        
        out6 = self.rrdl6(out5, out4)
        out7 = self.rrdl7(out6, out3)
        out8 = self.rrdl8(out7, out2)
        out9 = self.rrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10