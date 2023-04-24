# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:39:18 2023

@author: Gavin
"""

import numpy as np

from pconfig import IMG_SIZE, LUNA16_TRACHEA_FRACTION
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_dilation
)

def mask_luna16_ct(ct: np.ndarray) -> np.ndarray:
    
    lung_threshold = -700
    
    def mask_slide(slide):
        mask = slide < lung_threshold
        mask = remove_small_holes(mask, area_threshold=4096)
        
        labels = label(mask, background=0)
        
        sizes = [r.area for r in regionprops(labels)]
    
        largest_label = max(range(1, mask.max() + 1), key=lambda i: sizes[i-1])
        largest_label = labels == largest_label
        
        mask[largest_label] = 0

        labels = label(mask, background=0)
    
        cleaned_labels = remove_small_objects(
            labels,
            LUNA16_TRACHEA_FRACTION * 512**2,
            connectivity=1
        )
        
        mask = cleaned_labels > 0
        
        for _ in range(4): mask = binary_dilation(mask)
        
        mask = remove_small_holes(mask, area_threshold=1024)
        
        return mask
    
    masking_func = np.vectorize(mask_slide, signature='(n,m)->(n,m)')
    
    mask = masking_func(ct)
    
    return mask



def clean_luna16_ct(ct: np.ndarray) -> np.ndarray:
    
    lung_threshold = -700
    
    def mask_slide(slide):
        dupe_size = int(IMG_SIZE * 1.2)
        dupe = np.ones((dupe_size, dupe_size))
        
        mask = slide < lung_threshold
        
        lower = int(IMG_SIZE * 0.1)
        upper = int(IMG_SIZE * 1.1)
        
        dupe[lower:upper, lower:upper] = mask
        
        mask = dupe
        mask = label(mask)
        
        sizes = [r.area for r in regionprops(mask)]
    
        largest_label = max(range(1, mask.max() + 1), key=lambda i: sizes[i-1])
        
        mask = mask == largest_label
        
        for _ in range(8): mask = binary_dilation(mask)
        
        mask = remove_small_holes(mask, area_threshold=4096)
        mask = mask[lower:upper, lower:upper]
        
        return mask
        
    masking_func = np.vectorize(mask_slide, signature='(n,m)->(n,m)')
    
    mask = masking_func(ct)
    
    return mask



clean_luna16_cts = np.vectorize(clean_luna16_ct, otypes=[object])

mask_luna16_cts = np.vectorize(mask_luna16_ct, otypes=[object])