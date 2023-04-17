# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:55:27 2023

@author: Gavin
"""

import warnings, random

import numpy as np

from projectio import (
    load_nsclc_cts, 
    load_nsclc_segs, 
    plot_and_save_gif,
    save_instance
)
from pconfig import NSCLC_PREPROCESSED_DATA_DIR
from preprocessing import (
    clean_nsclc_cts, 
    mask_nsclc_cts, 
    apply_mask,
    apply_clahe,
    normalize,
    float64_to_cv8uc1,
    cv8uc1_to_float64,
    float64_to_float16,
    stack_channels,
    slides_filter,
    apply_filter
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    

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
        xs[2][:, :, :, 0],
        xs[2][:, :, :, 2], 
        xs[2][:, :, :, 1],
        ys[2]
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
    
    

if __name__ == '__main__':
    main()