# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:18:20 2023

@author: Gavin
"""

import torch

from torch import nn
from .dualr2wnet_fpn import DualR2WNetFPN
from .._core import BiFPNBlock

class DualR2WNetBiFPN(DualR2WNetFPN):

    def __init__(self, *args, bifpn_layers=2, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        backbone_kwargs = kwargs
        channels = backbone_kwargs['channels']

        self.bs = nn.Sequential(*[BiFPNBlock(channels) for _ in range(bifpn_layers)])

        for i in range(4):
            delattr(self, f'l{i + 1}')




    def forward(self, x):
        if len(x.shape) == 5:
            dim = 1
        elif len(x.shape) == 4:
            dim = 0

        fmaps = self.backbone(x)
        fmaps = [torch.cat((fmaps[i], fmaps[i + len(fmaps)]), dim=dim) for i in range(len(fmaps // 2))]

        fmaps[0] = self.a1(self.bn1(self.cn1(fmaps[0])))
        fmaps[1] = self.a2_1(self.bn2_1(self.cn2(fmaps[1])))
        fmaps[2] = self.a3_1(self.bn3_1(self.cn3(fmaps[2])))
        fmaps[3] = self.a4_1(self.bn4_1(self.cn4(fmaps[3])))

        fmaps = self.bs(fmaps)

        out1 = fmaps[0]
        out2 = self.a2_2(self.bn2_2(self.dcn2(fmaps[1])))
        out3 = self.a3_2(self.bn3_2(self.dcn3(fmaps[2])))
        out4 = self.a4_2(self.bn4_2(self.dcn4(fmaps[3])))

        out5 = self.a_out(self.cn_out(out1 + out2 + out3 + out4))

        return out5