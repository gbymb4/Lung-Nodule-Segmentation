# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:17:24 2023

@author: Gavin
"""

import torch

from torch import nn
from pconfig import OUT_DIR
from .dualr2unet import DualR2WNet

class DualR2UNetFE(nn.Module):
    
    def __init__(self, backbone) -> None:
        super().__init__()
        
        def backbone_forward(self, x):
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
                
            fmaps = [ 
                out10, out11, 
                out12, out13,
                out22, out23,
                out24, out25
            ]
            
            return fmaps
        
        backbone.foward = backbone_forward.__get__(backbone, DualR2WNet)
        
        self.backbone = backbone
        
    
    
    def forward(self, x):
        return self.backbone(x)
    


class DualR2UnetFPN(nn.Module):
    
    def __init__(self,
        pretrained_config=None, 
        **backbone_kwargs
    ) -> None:
        super().__init__()
        
        backbone = DualR2WNet(**backbone_kwargs)
        
        if pretrained_config is not None:
            pretrained_dataset = pretrained_config['dataset']
            pretrained_model_id = pretrained_config['model_id']
            
            if pretrained_config['root_dir'].lower() == 'na':
                pretrained_root_dir = f'{OUT_DIR}/{pretrained_dataset}/DualR2WNet'    
            else:
                pretrained_root_dir = pretrained_config['root_dir']
                
            pretrained_dir = f'{pretrained_root_dir}/{pretrained_model_id}'
            pretrained_model_fname = f'{pretrained_dir}/model'
            
            backbone.load_state_dict(torch.load(pretrained_model_fname))
            
        backbone = DualR2UNetFE(backbone)
        backbone.eval()
        
        self.backbone = backbone
        
        channels = backbone_kwargs['channels']
        
        self.dcn_1 = nn.ConvTranspose3d(
            channels * 8 * 2, 
            channels, 
            kernel_size=(1, 8, 8), 
            stride=(1, 8, 8), 
            padding=(0, 0, 0)
        )
        self.dcn_2 = nn.ConvTranspose3d(
            channels * 4 * 2, 
            channels, 
            kernel_size=(1, 4, 4), 
            stride=(1, 4, 4), 
            padding=(0, 0, 0)
        )
        self.dcn_3 = nn.ConvTranspose3d(
            channels * 2 * 2, 
            channels, 
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2), 
            padding=(0, 0, 0)
        )
        
        self.cn_1 = nn.Conv3d(
            channels * 2,
            channels, 
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.cn_2 = nn.Conv3d(
            channels,
            1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        
        self.a_1 = nn.ReLU()
        self.a_2 = nn.Sigmoid()
    
    
    def forward(self, x):
        if len(x.shape) == 5:
            dim = 1
        elif len(x.shape) == 4:
            dim = 0
        
        fmaps = self.backbone(x)
        
        out1 = self.dnc_1(torch.cat((fmaps[0] + fmaps[4]), dim=dim))
        out2 = self.dnc_2(torch.cat((fmaps[1] + fmaps[5]), dim=dim))
        out3 = self.dnc_3(torch.cat((fmaps[2] + fmaps[6]), dim=dim))
        out4 = self.cn_1(torch.cat((fmaps[3] + fmaps[7]), dim=dim))
        
        out5 = self.a_1(out1 + out2 + out3 + out4)
        out5 = self.a_2(self.cn_2(out5))
            
        return out5