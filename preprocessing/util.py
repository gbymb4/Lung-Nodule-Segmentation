# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:51:27 2023

@author: Gavin
"""

import numpy as np

def apply_mask(ct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ct_copy = ct.copy()
    ct_copy[~mask] = 0
    
    return ct_copy