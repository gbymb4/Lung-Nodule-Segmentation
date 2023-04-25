# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:54:35 2023

@author: Gavin
"""

from models.wnet import R2WNet

import torchsummary 

model = R2WNet(16, img_channels=3)

torchsummary.summary(model.cuda(), (3, 8, 360, 360), device='cuda')