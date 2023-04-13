# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:27:09 2023

@author: Gavin
"""

import numpy as np

from pconfig import IMG_SIZE, NSCLC_TRACHEA_FRACTION
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_dilation
)

def __mask_nsclc_ct(ct: np.ndarray) -> np.ndarray:
    
    lung_threshold = ct.min() + ((ct.max() - ct.min()) * 0.171)
    
    def mask_slide(slide):
        mask = slide < lung_threshold
        mask[slide == 0] = 0
        mask = remove_small_holes(mask, area_threshold=4096)
        
        labels = label(mask, background=0)
    
        cleaned_labels = remove_small_objects(
            labels,
            NSCLC_TRACHEA_FRACTION * 512**2,
            connectivity=1
        )
        
        mask = cleaned_labels > 0
        
        for _ in range(6): mask = binary_dilation(mask)
        
        mask = remove_small_holes(mask, area_threshold=1024)
        
        return mask
        
    masking_func = np.vectorize(mask_slide, signature='(n,m)->(n,m)')
    
    mask = masking_func(ct)
    
    return mask



def __clean_nsclc_ct(ct: np.ndarray) -> np.ndarray:
    
    lung_threshold = ct.min() + ((ct.max() - ct.min()) * 0.171)
    
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
        
        for _ in range(10): mask = binary_dilation(mask)
        
        mask = remove_small_holes(mask, area_threshold=4096)
        
        labels = label(~mask, background=0)

        if labels.max() == 2:
            props = regionprops(labels)
            
            centroid_y = [prop.centroid[0] for prop in props]
        
            max_index = np.argmax(centroid_y) + 1
        
            mask[labels == max_index] = 1
        
        mask = mask[lower:upper, lower:upper]
        
        return mask
        
    masking_func = np.vectorize(mask_slide, signature='(n,m)->(n,m)')
    
    mask = masking_func(ct)
    
    return mask



clean_nsclc_cts = np.vectorize(__clean_nsclc_ct, otypes=[object])

mask_nsclc_cts = np.vectorize(__mask_nsclc_ct, otypes=[object])