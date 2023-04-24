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
    lung_ct_mask = mask_nsclc_ct(ct)

    ct = apply_mask(ct, clean_ct_mask)
    
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
            
    
'''
def main():
    warnings.simplefilter('ignore')
    
    subsets = 2
    
    set_seed(0)
    
    normalize_vec = np.vectorize(normalize, otypes=[object])
    
    cts = load_nsclc_cts()
    cts = normalize_vec(cts)
    
    segs = load_nsclc_segs()

    clean_ct_masks = clean_nsclc_cts(cts)
    
    cleaner = np.vectorize(apply_mask, otypes=[object])
    cts = cleaner(cts, clean_ct_masks)
    
    lung_ct_masks = mask_nsclc_cts(cts)
    
    sfilter = np.vectorize(slides_filter, otypes=[object])
    filter_idxs = sfilter(lung_ct_masks)
    
    apply_sfilter = np.vectorize(apply_filter, otypes=[object])
    
    cts = apply_sfilter(cts, filter_idxs)
    segs = apply_sfilter(segs, filter_idxs)
    lung_ct_masks = apply_sfilter(lung_ct_masks, filter_idxs)

    float64_to_cv8uc1_vec = np.vectorize(float64_to_cv8uc1, otypes=[object])
    cv8uc1_to_float64_vec = np.vectorize(cv8uc1_to_float64, otypes=[object])
    float64_to_float16_vec = np.vectorize(float64_to_float16, otypes=[object])
    
    apply_clahe_vec = np.vectorize(apply_clahe, otypes=[object])
    
    clahe_cts = apply_clahe_vec(float64_to_cv8uc1_vec(cts))
    clahe_cts = cv8uc1_to_float64_vec(clahe_cts)
    clahe_cts = normalize_vec(clahe_cts)
    
    stack_channels_vec = np.vectorize(stack_channels, otypes=[object])
    
    xs = stack_channels_vec(cts, lung_ct_masks, clahe_cts)
    ys = segs
    
    gif_name = f'{NSCLC_PREPROCESSED_DATA_DIR}/example_mask.gif'
    
    to_plot = np.array([
        xs[0][:, :, :, 0],
        xs[0][:, :, :, 2], 
        xs[0][:, :, :, 1],
        ys[0]
    ], dtype=float)
    to_plot = to_plot.swapaxes(0, 1)
    
    titles = ['Clean Scan', 'CLAHE Scan', 'Lung Mask', 'Nodules Mask']
    
    plot_and_save_gif(to_plot, gif_name, titles=titles, verbose=True)
    
    xs = float64_to_float16_vec(xs)
    
    subset_idxs = np.random.permutation(len(xs))
    xs = xs[subset_idxs]
    ys = ys[subset_idxs]
    
    subsets_xs = np.array_split(xs, subsets)
    subsets_ys = np.array_split(ys, subsets)
    
    for i, (subset_xs, subset_ys) in enumerate(zip(subsets_xs, subsets_ys)):
        instance_idxs = np.arange(0, len(subset_xs))
        subset_num = np.full(len(subset_xs), i)
        dataset = np.full(len(subset_xs), 'NSCLC')
        
        print(f'saving subset {i + 1}...')
        
        saver = np.vectorize(save_instance, otypes=[object])
        saver(dataset, subset_num, instance_idxs, subset_xs, subset_ys)
'''
    

if __name__ == '__main__':
    main()