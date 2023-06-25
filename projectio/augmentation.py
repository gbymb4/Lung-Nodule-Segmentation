# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 16:09:36 2023

@author: Gavin
"""

import torch, random

import numpy as np
import torchvision.transforms.functional as F

def random_hflip(xs, ys, p=0.1):
    if np.random.rand() <= p:
        xs = xs.flip(-1)
        ys = ys.flip(-1)
        
    return xs, ys



def random_vflip(xs, ys, p=0.1):
    if np.random.rand() <= p:
        xs = xs.flip(-2)
        ys = ys.flip(-2)
        
    return xs, ys



def random_rotate(xs, ys, p=0.1):
    if np.random.rand() <= p:
        angle = random.uniform(0, 360)
        
        xs = F.rotate(xs, angle)
        ys = F.rotate(ys, angle)
        
    return xs, ys



def random_roll(xs, ys, max_shift=20, p=0.1):
    vert_roll = np.random.randint(0, max_shift)
    horiz_roll = np.random.randint(0, max_shift)
    
    if np.random.rand() <= p:
        xs = xs.roll(horiz_roll, -1)
        xs = xs.roll(vert_roll, -2)
        
        ys = ys.roll(horiz_roll, -1)
        ys = ys.roll(vert_roll, -2)
        
    return xs, ys