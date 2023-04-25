# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:15:47 2023

@author: Gavin
"""

import torch 

from torch import nn

class R2DBlock(nn.Module):
    
    def __init__(
        self, 
        channels
    ):
        super().__init__()
        
        self.cn1 = nn.Conv3d(
            channels, 
            channels, 
            (1, 3, 3), 
            (1, 1, 1), 
            (0, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.a1 = nn.ReLU()
        
        self.cn2 = nn.Conv3d(
            channels, 
            channels, 
            (1, 3, 3), 
            (1, 1, 1), 
            (0, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(channels)
        self.a2 = nn.ReLU()
        
        
        
    def forward(self, x):
        out1 = self.a1(self.bn1(self.cn1(x)))
        out2 = self.cn2(out1)
        
        out3 = self.bn2(out2 + x)
        out4 = self.a2(out3)
        
        return out4
    


class RR2DBlock(nn.Module):
    
    def __init__(
        self, 
        channels
    ):
        super().__init__()
        
        self.rcl1 = RCL2D(channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.a1 = nn.ReLU()
        
        self.rcl2 = RCL2D(channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.a2 = nn.ReLU()
        
        
        
    def forward(self, x):
        out1 = self.a1(self.bn1(self.rcl1(x)))
        out2 = self.rcl2(out1)
        
        out3 = self.bn2(out2 + x)
        out4 = self.a2(out3)
        
        return out4
    
    
    
class RREL2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__()
        
        if enc_ratio > 2:
            self.cn = nn.Conv3d(
                in_channels,
                out_channels, 
                kernel_size=(1, enc_ratio, enc_ratio), 
                padding=(0, 1, 1),
                stride=(1, enc_ratio, enc_ratio)
            )
        else: 
            k = max(enc_ratio, 3)
            
            self.cn = nn.Conv3d(
                in_channels,
                out_channels, 
                kernel_size=(1, k, k), 
                padding=(0, 1, 1),
                stride=(1, enc_ratio, enc_ratio)
            )
        
        self.a = nn.ReLU()
        self.b = RR2DBlock(out_channels)
        
    
    
    def forward(self, x):
        return self.b(self.a(self.cn(x)))



class RRDL2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, dec_ratio):
        super().__init__()
        
        if dec_ratio == 5:
            self.dcn = nn.ConvTranspose3d(
                in_channels,
                out_channels, 
                kernel_size=(1, 5, 5),
                padding=(0, 0, 0),
                stride=(1, 1, 1)
            )    
        else:
            self.dcn = nn.ConvTranspose3d(
                in_channels,
                out_channels, 
                kernel_size=(1, dec_ratio, dec_ratio),
                padding=(0, 0, 0),
                stride=(1, dec_ratio, dec_ratio)
            )
        
        self.a_1 = nn.ReLU()
        self.bn_1 = nn.BatchNorm3d(out_channels)
        self.cn = nn.Conv3d(
            out_channels * 2,
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        self.a_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm3d(out_channels)
        self.b = RR2DBlock(out_channels)
        
        
        
    def forward(self, x, skip):
        if len(x.shape) == 5:
            dim = 1
        elif len(x.shape) == 4:
            dim = 0
        
        out = torch.cat((self.bn_1(self.a_1(self.dcn(x))), skip), dim=dim)
        out = self.b(self.bn_2(self.a_2(self.cn(out))))

        return out
    


# based on implementation from https://github.com/TsukamotoShuchi/RCNN/blob/master/rcnnblock.py
class RCL2D(nn.Module):
    
    def __init__(self, channels, steps=4):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), 
            padding=(0, 1, 1), 
            bias=False
        )
        self.bn = nn.ModuleList([nn.BatchNorm3d(channels) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps

        self.shortcut = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), 
            padding=(0, 1, 1), bias=False
        )



    def forward(self, x):
        rx = x
        for i in range(self.steps):
            if i == 0:
                z = self.conv(x)
            else:
                z = self.conv(x) + self.shortcut(rx)
            x = self.relu(z)
            x = self.bn[i](x)
        return x
    


# based on implementation from https://github.com/TsukamotoShuchi/RCNN/blob/master/rcnnblock.py
class RCL3D(nn.Module):
    
    def __init__(self, channels, steps=4):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1), 
            bias=False
        )
        self.bn = nn.ModuleList([nn.BatchNorm3d(channels) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps

        self.shortcut = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1), bias=False
        )



    def forward(self, x):
        rx = x
        for i in range(self.steps):
            if i == 0:
                z = self.conv(x)
            else:
                z = self.conv(x) + self.shortcut(rx)
            x = self.relu(z)
            x = self.bn[i](x)
        return x