# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:48:47 2023

@author: Gavin
"""

import warnings, random

import numpy as np
import pylidc as pl
import scipy as sp
import skimage.io as io
import matplotlib.path as mpath

from skimage.morphology import remove_small_holes, binary_dilation
from projectio import (
    luna16_ct_fnames,
    load_luna16_ct, 
    luna16_seg_sids,
    load_luna16_seg, 
    load_luna16_censensus_contours,
    plot_and_save_gif,
    save_instance
)
from pconfig import (
    LUNA16_RAW_DATA_DIR,
    LUNA16_PREPROCESSED_DATA_DIR, 
    IMG_SIZE,
    CT_MAX,
    CT_MIN
)
from preprocessing import (
    mask_luna16_ct, 
    clean_luna16_ct,
    apply_mask,
    apply_clahe,
    normalize,
    float64_to_cv8uc1,
    cv8uc1_to_float64,
    float64_to_float32,
    float32_to_float64,
    stack_channels,
    slides_filter,
    apply_filter
)
from PIL import Image, ImageDraw
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)



def get_subset(idx, subset_idxs):
    matches = np.array([idx in idxs for idxs in subset_idxs])
    
    return np.argmax(matches)
    


def main():
    warnings.simplefilter('ignore')
    set_seed(0)
    
    train_frac = 0.7
    num_subsets = 4
    
    scans = pl.query(pl.Scan)
    all_ids = set(luna16_seg_sids())
    
    num_scans = len(all_ids)
    
    idxs = np.random.choice(num_scans, size=(num_scans,), replace=False)
    
    train_idxs = idxs[:int(num_scans * train_frac)]
    test_idxs = idxs[int(num_scans * train_frac):-1]
    
    subset_idxs = np.array_split(train_idxs, num_subsets)
    
    scan_count = 0
    for scan in tqdm(scans):
        series_instance_uid = scan.series_instance_uid
        num_slides = len(scan.slice_zvals)
        
        if series_instance_uid not in all_ids: continue
        
        imgs = scan.load_all_dicom_images(verbose=False)
        imgs = np.stack([dcm.pixel_array for dcm in imgs], axis=0)
        
        nodules_segs = np.zeros((num_slides, IMG_SIZE, IMG_SIZE), dtype=bool)
        nodules_contours = load_luna16_censensus_contours(scan)
        
        for nodule_contour in nodules_contours:
            
            for contour_slice in nodule_contour:
                contour_idx, contour_points = contour_slice
                contour_points = list(map(tuple, contour_points))
                
                slide = Image.new('L', [IMG_SIZE, IMG_SIZE], 0)
                ImageDraw.Draw(slide).polygon(contour_points, outline=1, fill=1)
                
                slide = np.array(slide, dtype=bool)
            
                nodules_segs[contour_idx] = np.logical_or(slide, nodules_segs[contour_idx])
            
        lungs_dir = f'{LUNA16_RAW_DATA_DIR}/lungs'
        
        lung_segs = io.imread(f'{lungs_dir}/{series_instance_uid}.mhd', plugin='simpleitk') 
        lung_segs = lung_segs.astype(bool)
        
        def grow_and_fill(slide):
            for _ in range(8): slide = binary_dilation(slide)
            
            slide = remove_small_holes(slide, area_threshold=4096)
            
            return slide
            
        grow_and_fill_lungs = np.vectorize(grow_and_fill, signature='(n,m)->(n,m)')
        lung_segs = grow_and_fill_lungs(lung_segs)
        
        imgs = imgs.astype(np.float32)
        
        imgs[imgs > CT_MAX] = CT_MAX
        imgs[imgs < CT_MIN] = CT_MIN
        imgs += -CT_MIN
        imgs /= (CT_MAX + -CT_MIN)
        
        if len(imgs) != len(lung_segs):
            imgs = imgs[len(imgs) - len(lung_segs) - 1:-1]
            nodules_segs = nodules_segs[len(imgs) - len(lung_segs) - 1:-1]
        
        #imgs[~lung_segs] = 0
        
        clahe_ct = apply_clahe(float64_to_cv8uc1(imgs))
        clahe_ct = cv8uc1_to_float64(clahe_ct)
        clahe_ct = normalize(clahe_ct)
        clahe_ct = float64_to_float32(clahe_ct)
        
        x = stack_channels(imgs, lung_segs, clahe_ct)
        y = nodules_segs
        
        if scan_count in test_idxs:
            save_instance('LUNA16', 'test', -1, scan_count, x, y)
            
        else:
            subset = get_subset(scan_count, subset_idxs)
            
            save_instance('LUNA16', 'train', subset, scan_count, x, y)
            
        if scan_count == 0:
            gif_name = f'{LUNA16_PREPROCESSED_DATA_DIR}/example_mask.gif'
            
            plot_x = float32_to_float64(x)
            
            to_plot = np.array([
                plot_x[:, :, :, 0],
                plot_x[:, :, :, 2], 
                plot_x[:, :, :, 1],
                y
            ], dtype=float)
            to_plot = to_plot.swapaxes(0, 1)
            
            titles = ['Scan', 'CLAHE Scan', 'Lung Mask', 'Nodules Mask']
            
            plot_and_save_gif(to_plot, gif_name, titles=titles, verbose=True)
        
        scan_count += 1
        
        
    
    
if __name__ == '__main__':
    main()