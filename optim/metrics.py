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



def sensitivity(pred, true):
    pred = (pred > 0.5).reshape(-1)
    true = true.bool().reshape(-1)
    
    true_positives = (pred & true).sum().item()
    positives = true.sum().item()
    
    return true_positives / positives if positives > 0 else 0



def specificity(pred, true):
    pred = (pred > 0.5).reshape(-1)
    true = true.bool().reshape(-1)
    
    true_negatives = (~pred & ~true).sum().item()
    negatives = (~true).sum().item()
    
    return true_negatives / negatives if negatives > 0 else 0
    


def fpr(pred, true):
    pred = (pred > 0.5).reshape(-1)
    true = true.bool().reshape(-1)
    
    false_positives = (pred & ~true).sum().item()
    negatives = (~true).sum().item()

    return false_positives / negatives if negatives > 0 else 0



def hard_dice(pred, true, epsilon=1e-7):
    criterion = HardDiceLoss(epsilon=epsilon)
    
    return criterion(pred, true).item()



def compute_all_metrics(pred, true, epsilon=1-7):
    results = {}
    
    results['accuracy'] = accuracy(pred, true)
    results['sensitivity'] = sensitivity(pred, true)
    results['specificity'] = specificity(pred, true)
    results['hard_dice'] = hard_dice(pred, true, epsilon=1-7)
    
    return results