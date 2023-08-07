# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:56:20 2023

@author: Gavin
"""

import numpy as np

from scipy.ndimage import label, center_of_mass

def compute_centroids(nodules_mask):
    binary_np = nodules_mask.detach().cpu().numpy()

    labeled_np, num_features = label(binary_np)

    centroids = []
    for label_id in range(1, num_features + 1):
        labeled_region = (labeled_np == label_id).astype(np.float32)
        centroid_z, centroid_y, centroid_x = center_of_mass(labeled_region)
        centroids.append((centroid_z, centroid_y, centroid_x))

    return centroids



def centroid_slides(tensor, centroids):
    zs = [round(z) for z, _, _ in centroids]
    slides = tensor[zs]
    
    return slides