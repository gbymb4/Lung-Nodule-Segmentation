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
    


class RR3DBlock(nn.Module):
    
    def __init__(
        self, 
        channels
    ):
        super().__init__()
        
        self.rcl1 = RCL3D(channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.a1 = nn.ReLU()
        
        self.rcl2 = RCL3D(channels)
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
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU()
        b = RR2DBlock(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
        
    
    
    def forward(self, x):
        return self.features(x)
    


class DRREL(nn.Module):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__()
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU()
        
        self.b2d = RR2DBlock(out_channels)
        self.b3d = RR3DBlock(out_channels)
        
        layers.extend((cn, bn, a))
        
        self.backbone = nn.Sequential(*layers)
        
    
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.b2d(out) + self.b3d(out)
        
        return out
    
    

class RRDL2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, dec_ratio):
        super().__init__()
        
        if dec_ratio == 3:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 3, 3),
                padding=(0, 0, 0)
            )
        elif dec_ratio == 2:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1)
            )
        else:
            raise ValueError('dec_ratio other than 2 or 3 is not supported')
        
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
    
    

class DRRDL(nn.Module):
    
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
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        
        self.a_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm3d(out_channels)
        
        self.b_1 = RR2DBlock(out_channels)
        self.b_2 = RR3DBlock(out_channels)
        
        
        
    def forward(self, x, skip):
        if len(x.shape) == 5:
            dim = 1
        elif len(x.shape) == 4:
            dim = 0
        
        out1 = torch.cat((self.bn_1(self.a_1(self.dcn(x))), skip), dim=dim)
        out1 = self.bn_2(self.a_2(self.cn(out1)))
        
        out2 = self.b_1(out1)
        out3 = self.b_2(out1)
        
        out4 = out2 + out3

        return out4
    


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


class BiFPNBlock(nn.Module):

    def __init__(self, channels, size=360):
        self.cn1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.cn1_a = nn.ReLU()
        self.cn1_bn = nn.BatchNorm3d(channels)
        self.l1_cn = nn.Sequential(self.cn1, self.cn1_bn, self.cn1_a)

        self.ds1 = nn.Conv3d(
            channels,
            channels * 2,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.ds1_a = nn.ReLU()
        self.ds1_bn = nn.BatchNorm3d(channels * 2)
        self.l1_ds = nn.Sequential(self.ds1, self.ds1_bn, self.ds1_a)
        self.us1 = nn.ConvTranspose3d(
            channels * 2,
            channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )
        self.us1_a = nn.ReLU()
        self.us1_bn = nn.BatchNorm3d(channels)
        self.l1_us = nn.Sequential(self.us1, self.us1_bn, self.us1_a)

        self.cn2_1 = nn.Conv3d(
            channels * 2,
            channels * 2,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.cn2_1_a = nn.ReLU()
        self.cn2_1_bn = nn.BatchNorm3d(channels * 2)
        self.l2_1_cn = nn.Sequential(self.cn2_1, self.cn2_1_bn, self.cn2_1_a)
        self.cn2_2 = nn.Conv3d(
            channels * 2,
            channels * 2,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.cn2_2_a = nn.ReLU()
        self.cn2_2_bn = nn.BatchNorm3d(channels * 2)
        self.l2_2_cn = nn.Sequential(self.cn2_2, self.cn2_2_bn, self.cn2_2_a)

        self.ds2 = nn.Conv3d(
            channels * 2,
            channels * 4,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.ds2_a = nn.ReLU()
        self.ds2_bn = nn.BatchNorm3d(channels * 4)
        self.l2_ds = nn.Sequential(self.ds2, self.ds2_bn, self.ds2_a)
        self.us2 = nn.ConvTranspose3d(
            channels * 4,
            channels * 2,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )
        self.us2_a = nn.ReLU()
        self.us2_bn = nn.BatchNorm3d(channels * 2)
        self.l2_us = nn.Sequential(self.us2, self.us2_bn, self.us2_a)

        self.cn3_1 = nn.Conv3d(
            channels * 4,
            channels * 4,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.cn3_1_a = nn.ReLU()
        self.cn3_1_bn = nn.BatchNorm3d(channels * 4)
        self.l3_1_cn = nn.Sequential(self.cn3_1, self.cn3_1_bn, self.cn3_1_a)
        self.cn3_2 = nn.Conv3d(
            channels * 4,
            channels * 4,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.cn3_2_a = nn.ReLU()
        self.cn3_2_bn = nn.BatchNorm3d(channels * 4)
        self.l3_2_cn = nn.Sequential(self.cn3_2, self.cn3_2_bn, self.cn3_2_a)

        self.ds3 = nn.Conv3d(
            channels * 4,
            channels * 8,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.ds3_a = nn.ReLU()
        self.ds3_bn = nn.BatchNorm3d(channels * 8)
        self.l3_ds = nn.Sequential(self.ds3, self.ds3_bn, self.ds3_a)
        self.us3 = nn.ConvTranspose3d(
            channels * 8,
            channels * 4,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )
        self.us3_a = nn.ReLU()
        self.us3_bn = nn.BatchNorm3d(channels * 4)
        self.l3_us = nn.Sequential(self.us3, self.us3_us, self.us3_a)

        self.cn4 = nn.Conv3d(
            channels * 8,
            channels * 8,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.cn4_a = nn.ReLU()
        self.cn4_bn = nn.BatchNorm3d(channels * 8)
        self.l4_cn = nn.Sequential(self.cn4, self.cn4_bn, self.cn4_a)

        self.l1_in_w = nn.Parameter(torch.rand(1))
        self.l2_in_w = nn.Parameter(torch.rand(1))
        self.l3_in_w = nn.Parameter(torch.rand(1))
        self.l4_in_w = nn.Parameter(torch.rand(1))

        self.l1_ds_w = nn.Parameter(torch.rand(1))
        self.l2_ds_w = nn.Parameter(torch.rand(1))
        self.l3_ds_w = nn.Parameter(torch.rand(1))

        self.l1_us_w = nn.Parameter(torch.rand(1))
        self.l2_us_w = nn.Parameter(torch.rand(1))
        self.l3_us_w = nn.Parameter(torch.rand(1))

        self.l1_ds_w = nn.Parameter(torch.rand(1))
        self.l2_ds_w = nn.Parameter(torch.rand(1))
        self.l3_ds_w = nn.Parameter(torch.rand(1))

        self.out2_1_w = nn.Parameter(torch.rand(1))
        self.out3_1_w = nn.Parameter(torch.rand(1))

        self.out2_2_w = nn.Parameter(torch.rand(1))
        self.out3_2_w = nn.Parameter(torch.rand(1))

        self.out1_w = nn.Parameter(torch.rand(1))
        self.out4_w = nn.parameter(torch.rand(1))

        self.e = 1e-4


    def forward(self, fmaps):
        l1_in, l2_in, l3_in, l4_in = fmaps

        out2_1 = self.l2_1_cn(
            (self.l2_in_w * l2_in + self.l1_ds_w * self.l1_ds(l1_in)) /
            (self.l2_in_w + self.l1_ds_w + self.e)
        )
        out3_1 = self.l3_1_cn(
            (self.l3_in_w * l3_in + self.l2_ds_w + self.l2_ds(out2_1)) /
            (self.l3_in_w + self.l2_ds_w + self.e)
        )

        out4 = self.l4_cn(
            (self.l4_in_w * l4_in + self.l3_ds * self.l3_ds(out3_1)) /
            (self.l4_in_w + self.l3_ds + self.e)
        )

        out3_2 = self.l3_2_cn(
            (self.l3_us_w * self.l3_us(out4) + self.out3_1_w * out3_1 + self.l3_in_w * l3_in) /
            (self.l3_us_w + self.out3_1_w + self.l3_in_w + self.e)
        )
        out2_2 = self.l2_2_cn(
            (self.l2_us_w * self.l2_us(out3_2) + self.out_2_1_w * out2_1 + self.l2_in_w * l2_in) /
            (self.l2_us_w + self.out_2_1_w + self.l2_in_w + self.e)
        )

        out1 = self.l1_cn(
            (self.l1_us_w * self.l1_us(out2_2) + self.l1_in_w * l1_in) /
            (self.l1_us_w + self.l1_in_w + self.e)
        )

        out = [out1, out2_2, out3_2, out4]

        return out

