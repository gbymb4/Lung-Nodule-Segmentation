# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:48:47 2023

@author: Gavin
"""

import warnings

import numpy as np

from projectio import load_luna16_cts, plot_and_save_gif
from preprocessing import mask_luna16_cts, apply_mask
from pconfig import LUNA16_PREPROCESSED_DATA_DIR

def main():
    warnings.simplefilter('ignore')
    
    data = load_luna16_cts(0)
    
    masks = mask_luna16_cts(data)
    
    apply_mask_vec = np.vectorize(apply_mask, otypes=[object])
    masked_cts = apply_mask_vec(data, masks)
    
    gif_name = f'{LUNA16_PREPROCESSED_DATA_DIR}/example_mask.gif'
    
    to_plot = np.array([data[0], masks[0], abs(masked_cts[0])], dtype=float)
    to_plot = to_plot.swapaxes(0, 1)
    
    titles = ['Raw Scan', 'Lung Mask', 'Masked Scan']
    
    plot_and_save_gif(to_plot, gif_name, titles=titles, verbose=True)
    
    
    
if __name__ == '__main__':
    main()