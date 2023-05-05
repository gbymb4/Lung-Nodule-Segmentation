# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:18:05 2023

@author: Gavin
"""

import torch

from torch import nn
from .dualr2unet_fpn import DualR2UNetFPN
from .._core import BiFPNBlock

class DualR2UNetBiFPN(DualR2UNetFPN):

    def __init__(self, *args, bifpn_layers=2, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        backbone_kwargs = kwargs
        channels = backbone_kwargs['channels']

        self.bs = nn.Sequential(*[BiFPNBlock(channels) for _ in range(bifpn_layers)])



    def forward(self, x):
        if len(x.shape) == 5:
            dim = 1
        elif len(x.shape) == 4:
            dim = 0

        fmaps = self.backbone(x)
        fmaps = self.bs(fmaps)

        out1 = fmaps[0]
        out2 = self.a2(self.bn2(self.dnc2(fmaps[1])))
        out3 = self.a3(self.bn3(self.dnc3(fmaps[2])))
        out4 = self.a4(self.bn4(self.dnc4(fmaps[4])))

        out5 = self.a_out(self.cn_out(out1 + out2 + out3 + out4))

        return out5