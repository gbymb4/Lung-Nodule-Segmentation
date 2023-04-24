# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:48:47 2023

@author: Gavin
"""

import warnings

import numpy as np

from projectio import (
    luna16_ct_fnames,
    load_luna16_ct, 
    luna16_seg_subset_sids,
    load_luna16_seg, 
    plot_and_save_gif,
    save_instance
)
from pconfig import LUNA16_PREPROCESSED_DATA_DIR
from preprocessing import (
    mask_luna16_ct, 
    clean_luna16_ct,
    apply_mask,
    apply_clahe,
    normalize,
    float64_to_cv8uc1,
    cv8uc1_to_float64,
    float64_to_float16,
    float16_to_float64,
    stack_channels,
    slides_filter,
    apply_filter
)

def preprocess_ct(fname, sid):
    ct = load_luna16_ct(fname)
    seg = load_luna16_seg(sid)
    
    clean_ct_mask = clean_luna16_ct(ct)
    lung_ct_mask = mask_luna16_ct(ct)

    ct = apply_mask(ct, clean_ct_mask)
    
    filter_idxs = slides_filter(lung_ct_mask)
    
    ct = apply_filter(ct, filter_idxs)
    seg = apply_filter(seg, filter_idxs)
    lung_ct_mask = apply_filter(lung_ct_mask, filter_idxs)
    
    ct = normalize(ct)
    
    clahe_ct = apply_clahe(float64_to_cv8uc1(ct))
    clahe_ct = cv8uc1_to_float64(clahe_ct)
    clahe_ct = normalize(clahe_ct)
    clahe_ct = float64_to_float16(clahe_ct)
    
    x = stack_channels(ct, lung_ct_mask, clahe_ct)
    x = float64_to_float16(x)
    
    y = seg
    
    return x, y
    
    

def main():
    warnings.simplefilter('ignore')
    
    for subset in range(10):
        ct_fnames = luna16_ct_fnames(subset)
        seg_sids = luna16_seg_subset_sids(subset)
        
        for i, (ct_fname, seg_sid) in enumerate(zip(ct_fnames, seg_sids)):
            x, y = preprocess_ct(ct_fname, seg_sid)
            
            if subset == 0 and i == 0:
                gif_name = f'{LUNA16_PREPROCESSED_DATA_DIR}/example_mask.gif'
                
                plot_x = float16_to_float64(x)
                
                to_plot = np.array([
                    plot_x[:, :, :, 0],
                    plot_x[:, :, :, 2], 
                    plot_x[:, :, :, 1],
                    y
                ], dtype=float)
                to_plot = to_plot.swapaxes(0, 1)
                
                titles = ['Clean Scan', 'CLAHE Scan', 'Lung Mask', 'Nodules Mask']
                
                plot_and_save_gif(to_plot, gif_name, titles=titles, verbose=True)
            
            print(f'saving subset {subset + 1} instance {i + 1}...', end='')
            
            save_instance('LUNA16', subset, i, x, y)
            
            print('done')
            
    
    
if __name__ == '__main__':
    main()