# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:17:54 2023

@author: Gavin
"""

import torch

from torch import nn
from pconfig import OUT_DIR
from .dualr2unet import DualR2UNet

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
                
            fmaps = [ 
                out10, out11, 
                out12, out13
            ]
            
            return fmaps
        
        backbone.foward = backbone_forward.__get__(backbone, DualR2UNet)
        
        self.backbone = backbone
        
    
    
    def forward(self, x):
        return self.backbone(x)
    


class DualR2UnetFPN(nn.Module):
    
    def __init__(self,
        pretrained_config=None, 
        **backbone_kwargs
    ) -> None:
        super().__init__()
        
        backbone = DualR2UNet(**backbone_kwargs)
        
        if pretrained_config is not None:
            pretrained_dataset = pretrained_config['dataset']
            pretrained_model_id = pretrained_config['model_id']
            
            if pretrained_config['root_dir'].lower() == 'na':
                pretrained_root_dir = f'{OUT_DIR}/{pretrained_dataset}/DualR2UNet'    
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
            channels * 8, 
            channels, 
            kernel_size=(1, 8, 8), 
            stride=(1, 8, 8), 
            padding=(0, 0, 0)
        )
        self.dcn_2 = nn.ConvTranspose3d(
            channels * 4, 
            channels, 
            kernel_size=(1, 4, 4), 
            stride=(1, 4, 4), 
            padding=(0, 0, 0)
        )
        self.dcn_3 = nn.ConvTranspose3d(
            channels * 2, 
            channels, 
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2), 
            padding=(0, 0, 0)
        )
        
        self.cn = nn.Conv3d(
            channels,
            1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        self.a = nn.Sigmoid()
    
    
    def forward(self, x):
        fmaps = self.backbone(x)
        
        out1 = self.dnc_1(fmaps[0])
        out2 = self.dnc_2(fmaps[1])
        out3 = self.dnc_3(fmaps[2])
        out4 = fmaps[3]
        
        out5 = out1 + out2 \
            + out3 + out4
        out5 = self.a(self.cn(out5))
            
        return out5