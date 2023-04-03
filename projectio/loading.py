# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:51:42 2023

@author: Gavin
"""

import os

import numpy as np
import skimage.io as io

from pconfig import LUNA16_RAW_DATA_DIR

def load_luna16_cts(subset: int) -> np.ndarray:
    data_dir = f'{LUNA16_RAW_DATA_DIR}/subset{subset}'
    
    all_headers = list(filter(lambda x: x[-4:] == '.mhd', os.listdir(data_dir)))
    all_headers = np.array([f'{data_dir}/{elem}' for elem in all_headers], dtype=object)

    loader = np.vectorize(load_luna16_ct, otypes=[object])

    all_cts = loader(all_headers[:2])
    
    return all_cts



def load_luna16_ct(fname: str) -> np.ndarray:
    ct = io.imread(fname, plugin='simpleitk')  
    
    return ct