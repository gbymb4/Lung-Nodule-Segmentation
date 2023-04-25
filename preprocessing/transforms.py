# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:14:08 2023

@author: Gavin
"""

import numpy as np

from pconfig import IMG_SIZE
from typing import Tuple
from scipy.ndimage import zoom

def zoom_and_resize_ct(
    ct: np.ndarray, 
    seg: np.ndarray,
    new_size: Tuple[int, ...]=(360, 360)
) -> np.ndarray:
    
    def slide_min_padding(slide):
        nonzero_indices = np.nonzero(slide > 0)
        
        if any([arr.shape == (0,) for arr in nonzero_indices]): 
            return IMG_SIZE
        
        bot_row_padding = IMG_SIZE - np.amax(nonzero_indices[0])
        top_row_padding = np.amin(nonzero_indices[0])
        
        right_col_padding = IMG_SIZE - np.amax(nonzero_indices[1])
        left_col_padding = np.amin(nonzero_indices[1])
        
        return min([bot_row_padding, top_row_padding, right_col_padding, left_col_padding])

    slide_min_padding_vec = np.vectorize(slide_min_padding, signature='(n,m)->()')
    slide_min_paddings = slide_min_padding_vec(ct)

    min_padding = min(slide_min_paddings)
    crop_area = (IMG_SIZE - min_padding) ** 2

    zoom_factor = ((IMG_SIZE**2) / (crop_area))
    resize_factor = (new_size[0] / IMG_SIZE, new_size[1] / IMG_SIZE)

    def zoom_and_resize(slide, seg_slide):
        zoomed_slide = zoom(slide, zoom_factor, order=0)
        zoomed_seg = zoom(seg_slide, zoom_factor, order=0)
        
        center_i = zoomed_slide.shape[0] // 2
        center_j = zoomed_slide.shape[1] // 2
        
        crop_pad = IMG_SIZE // 2
        
        zoomed_slide = zoomed_slide[center_i-crop_pad:center_i+crop_pad,
                                    center_j-crop_pad:center_j+crop_pad]
        zoomed_seg = zoomed_seg[center_i-crop_pad:center_i+crop_pad,
                                center_j-crop_pad:center_j+crop_pad]
        
        resized_slide = zoom(zoomed_slide, resize_factor, order=0)
        resized_seg = zoom(zoomed_seg, resize_factor, order=0)
        
        return resized_slide, resized_seg
        
    zoom_and_resize_vec = np.vectorize(zoom_and_resize, signature='(n,m),(n,m)->(i,j),(i,j)')
    zoomed_ct_and_seg = zoom_and_resize_vec(ct)
    
    return zoomed_ct_and_seg