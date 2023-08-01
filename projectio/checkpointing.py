# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:24:57 2023

@author: Gavin
"""

import os, json, torch

import numpy as np

from pconfig import OUT_DIR

def last_checkpoint(dataset, model, root_id, train_idx):
    out_root = f'{OUT_DIR}/{dataset.lower()}/{type(model).__name__}/{root_id}'
    
    if not os.path.isdir(out_root):
        return 0, None, None
    
    runs = os.listdir(out_root)
    run_train_idxs = np.array([int(r.split('_')[0]) for r in runs])
    
    if train_idx not in run_train_idxs:
        return 0, None, None
    
    run_idx = np.argmax(run_train_idxs == train_idx)
    run_with_checkpoints = runs[run_idx]
    
    checkpoints_dir = f'{out_root}/{run_with_checkpoints}/model_checkpoints'
    
    if not os.path.isdir(checkpoints_dir):
        return 0, None, None
    
    model_checkpoints = os.listdir(checkpoints_dir)
    model_checkpoints = sorted(
        model_checkpoints, 
        key=lambda x: int(x[6:-3])
    )
    
    latest_epoch =  int(model_checkpoints[-1][6:-3])
    latest_id = int(run_with_checkpoints.split('_')[1])
    latest_checkpoint = f'{checkpoints_dir}/{model_checkpoints[-1]}'
    
    model.load_state_dict(torch.load(latest_checkpoint))
    
    with open(f'{out_root}/{run_with_checkpoints}/history.json') as f:
        latest_history = json.load(f)
    
    return latest_epoch, latest_id, latest_history