# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:51:27 2023

@author: Gavin
"""

import cv2

import numpy as np

from typing import Tuple

def apply_mask(ct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ct_copy = ct.copy()
    ct_copy[mask] = ct.min()
    
    return ct_copy



def apply_clahe(ct: np.ndarray) -> np.ndarray:
    transform = cv2.createCLAHE()
    transform_vec = np.vectorize(transform.apply, signature='(n,m)->(n,m)')
    
    clahe = transform_vec(ct)
    
    return clahe



def normalize(array: np.ndarray) -> np.ndarray:
    min_value = np.min(array)
    max_value = np.max(array)
    
    norm_array = (array - min_value) / (max_value - min_value)
    
    return norm_array



def float64_to_cv8uc1(array: np.ndarray) -> np.ndarray:
    convert = (array * 255).astype(np.uint8)
    
    return convert
    


def cv8uc1_to_float64(array: np.ndarray) -> np.ndarray:
    convert = (array / 255).astype(np.float64)
    
    return convert



def float64_to_float16(array: np.ndarray) -> np.ndarray:
    convert = array.astype(np.float16)
    
    return convert



def stack_channels(*arrays: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.stack(arrays, axis=-1)