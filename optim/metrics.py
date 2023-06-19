# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:26:47 2023

@author: Gavin
"""

import torch

from .loss import HardDiceLoss

def accuracy(pred, true):
    pred = (pred > 0.5).reshape(-1)
    true = true.reshape(-1)
    
    correct = torch.sum(pred == true).item()
    total = len(pred)
    
    return correct / total



def sensitivity(pred, true, epsilon=1e-7):
    pred = (pred > 0.5).reshape(-1)
    true = true.reshape(-1)
    
    true_positives = torch.sum(torch.logical_and(pred, true)).item()
    positives = torch.sum(true).item()
    
    return true_positives / (positives + epsilon)



def specificity(pred, true, epsilon=1e-7):
    pred = (pred > 0.5).reshape(-1)
    true = true.reshape(-1)
    
    true_negatives = torch.sum(torch.logical_and(~pred, ~true)).item()
    negatives = torch.sum(~true).item()
    
    return true_negatives / (negatives + epsilon)
    


def hard_dice(pred, true, epsilon=1e-7):
    criterion = HardDiceLoss(epsilon=epsilon)
    
    return criterion(pred, true).item()



def compute_all_metrics(pred, true, epsilon=1-7):
    results = {}
    
    results['accuracy'] = accuracy(pred, true)
    results['sensitivity'] = sensitivity(pred, true, epsilon=1-7)
    results['specificity'] = specificity(pred, true, epsilon=1-7)
    results['hard_dice'] = hard_dice(pred, true, epsilon=1-7)
    
    return results