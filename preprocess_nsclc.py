# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:55:27 2023

@author: Gavin
"""

import warnings, random

import numpy as np

from projectio import (
    nsclc_ct_fnames,
    load_nsclc_ct, 
    nsclc_seg_fnames,
    load_nsclc_seg, 
    plot_and_save_gif,
    save_instance
)
from pconfig import NSCLC_PREPROCESSED_DATA_DIR
from preprocessing import (
    clean_nsclc_ct, 
    mask_nsclc_ct, 
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    


def preprocess_ct(ct_fname, seg_fname):
    ct = load_nsclc_ct(ct_fname)
    ct = normalize(ct)
    
    seg = load_nsclc_seg(seg_fname)
    
    clean_ct_mask = clean_nsclc_ct(ct)
    ct = apply_mask(ct, clean_ct_mask)
    
    lung_ct_mask = mask_nsclc_ct(ct)
    
    filter_idxs = slides_filter(lung_ct_mask)
    
    ct = apply_filter(ct, filter_idxs)
    seg = apply_filter(seg, filter_idxs)
    lung_ct_mask = apply_filter(lung_ct_mask, filter_idxs)
    
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
    
    subsets = 5
    
    set_seed(0)
    
    ct_fnames = nsclc_ct_fnames()
    seg_fnames = nsclc_seg_fnames()
    
    subset_idxs = np.random.permutation(len(ct_fnames))
    ct_fnames = ct_fnames[subset_idxs]
    seg_fnames = seg_fnames[subset_idxs]
    
    subsets_ct_fnames = np.array_split(ct_fnames, subsets)
    subsets_seg_fnames = np.array_split(seg_fnames, subsets)
    
    for i, (subset_ct_fnames, subset_seg_fnames) in enumerate(zip(subsets_ct_fnames, subsets_seg_fnames)):
        subset = i
        
        for j, (ct_fname, seg_fname) in enumerate(zip(subset_ct_fnames, subset_seg_fnames)):
            x, y = preprocess_ct(ct_fname, seg_fname)
        
            if subset == 0 and j == 0:
                gif_name = f'{NSCLC_PREPROCESSED_DATA_DIR}/example_mask.gif'
                
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
        
            print(f'saving subset {subset + 1} instance {j + 1}...', end='')
            
            save_instance('NSCLC', subset, j, x, y)
            
            print('done')
            
    

if __name__ == '__main__':
    main()