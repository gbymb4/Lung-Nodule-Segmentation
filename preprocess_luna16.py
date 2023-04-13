# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:48:47 2023

@author: Gavin
"""

import warnings

import numpy as np

from projectio import load_luna16_cts, load_luna16_segs, plot_and_save_gif
from pconfig import LUNA16_PREPROCESSED_DATA_DIR
from preprocessing import (
    mask_luna16_cts, 
    clean_luna16_cts,
    apply_mask,
    apply_clahe,
    normalize,
    float64_to_cv8uc1,
    cv8uc1_to_float64,
    float64_to_float16,
    stack_channels
)

def main():
    warnings.simplefilter('ignore')
    
    cts = load_luna16_cts(0)
    segs = load_luna16_segs(0)
    
    clean_ct_masks = clean_luna16_cts(cts)
    
    cleaner = np.vectorize(apply_mask, otypes=[object])
    cts = cleaner(cts, clean_ct_masks)
    
    lung_ct_masks = mask_luna16_cts(cts)
    
    float64_to_cv8uc1_vec = np.vectorize(float64_to_cv8uc1, otypes=[object])
    cv8uc1_to_float64_vec = np.vectorize(cv8uc1_to_float64, otypes=[object])
    float64_to_float16_vec = np.vectorize(float64_to_float16, otypes=[object])
    
    apply_clahe_vec = np.vectorize(apply_clahe, otypes=[object])
    normalize_vec = np.vectorize(normalize, otypes=[object])
    
    cts = normalize_vec(cts)
    
    clahe_cts = apply_clahe_vec(float64_to_cv8uc1_vec(cts))
    clahe_cts = cv8uc1_to_float64_vec(clahe_cts)
    clahe_cts = normalize_vec(clahe_cts)
    
    stack_channels_vec = np.vectorize(stack_channels, otypes=[object])
    
    xs = stack_channels_vec(cts, lung_ct_masks, clahe_cts)
    ys = segs
    
    gif_name = f'{LUNA16_PREPROCESSED_DATA_DIR}/example_mask.gif'
    
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
    
    
    
if __name__ == '__main__':
    main()