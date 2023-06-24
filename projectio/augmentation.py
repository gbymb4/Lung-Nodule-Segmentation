# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 16:09:36 2023

@author: Gavin
"""

import torch, random

import numpy as np
import torch.nn.functional as F

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
    device = xs.device
    
    def rotate(tensor, angle):
        *dims, h, w = tensor.size()
    
        grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h))
        grid = torch.stack((grid_x.float(), grid_y.float()), dim=-1)
    
        center = torch.tensor([w / 2, h / 2]).float()
        
        rotation_matrix = torch.tensor(
            [[torch.cos(angle), -torch.sin(angle)],
             [torch.sin(angle), torch.cos(angle)]],
        ).float()
        rotated_grid = torch.matmul(grid - center, rotation_matrix) + center
    
        normalized_grid = rotated_grid / torch.tensor([w - 1, h - 1]).float()
        normalized_grid = normalized_grid.view(-1, 2)
    
        rotated_tensor = F.grid_sample(
            tensor, 
            normalized_grid.view(1, 1, -1, 2), 
            align_corners=True
        )
    
        rotated_tensor = rotated_tensor.reshape(*dims, h, w)
    
        return rotated_tensor.to(device)
    
    if np.random.rand() <= p:
        angle = random.uniform(0, 360)
        
        xs = rotate(xs, angle)
        ys = rotate(ys, angle)
        
    return xs, 



def random_roll(xs, ys, max_shift=20, p=0.1):
    vert_roll = np.random.randint(0, max_shift)
    horiz_roll = np.random.randint(0, max_shift)
    
    if np.random.rand() <= p:
        xs = xs.roll(horiz_roll, -1)
        xs = xs.roll(vert_roll, -2)
        
        ys = ys.roll(horiz_roll, -1)
        ys = ys.roll(vert_roll, -2)
        
    return xs, ys