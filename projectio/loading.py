# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:51:42 2023

@author: Gavin
"""

import os

import numpy as np
import skimage.io as io
import pydicom as dicom
import pylidc as pl
import matplotlib.path as mpath

from pconfig import LUNA16_RAW_DATA_DIR, NSCLC_RAW_DATA_DIR, IMG_SIZE
from preprocessing import normalize

def load_luna16_cts(subset: int) -> np.ndarray:
    data_dir = f'{LUNA16_RAW_DATA_DIR}/subset{subset}'
    
    all_headers = list(filter(lambda x: x[-4:] == '.mhd', os.listdir(data_dir)))
    all_headers = np.array([f'{data_dir}/{elem}' for elem in all_headers], dtype=object)

    def load_luna16_ct(fname: str) -> np.ndarray:
        ct = io.imread(fname, plugin='simpleitk')  
        
        return ct

    loader = np.vectorize(load_luna16_ct, otypes=[object])

    all_cts = loader(all_headers[:3])
    
    return all_cts



def load_luna16_segs(subset: int) -> np.ndarray:
    data_dir = f'{LUNA16_RAW_DATA_DIR}/subset{subset}'
    
    all_headers = list(filter(lambda x: x[-4:] == '.mhd', os.listdir(data_dir)))
    subset_sids = np.array([header[:-4] for header in all_headers], dtype=object)
    
    scans = pl.query(pl.Scan)
    
    filtered_scans = {scan.series_instance_uid: scan for scan in scans if scan.
                          series_instance_uid in subset_sids}

    def load_luna16_seg(sid: str) -> np.ndarray:
        scan = filtered_scans[sid]
        annotations = np.array(scan.annotations, dtype=object)
        
        num_slices = len(scan.slice_zvals)
        
        masks = np.zeros((num_slices, IMG_SIZE, IMG_SIZE))
        
        def annotation_mask(annotation: pl.Annotation) -> None:
            idxs = np.array(annotation.contour_slice_indices)
            contours_matrix = annotation.contours_matrix
            
            def annotation_mask_slice(slice_idx: int) -> None:
                points = contours_matrix[contours_matrix[:, 2] == slice_idx][:, :2]
                points = np.array(points)
                
                rr, cc = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE), indexing='ij')
                mask = np.logical_and(np.invert(np.isnan(rr)), np.invert(np.isnan(cc)))
                mask = np.array([rr.flatten(), cc.flatten()]).T

                path = mpath.Path(points)
                
                slide = masks[slice_idx]

                slide[mask[:, 0], mask[:, 1]] = path.contains_points(mask)
                
            annotation_mask_slice_vec = np.vectorize(annotation_mask_slice)
            annotation_mask_slice_vec(idxs)
            
        annotation_mask_vec = np.vectorize(annotation_mask, otypes=[object])
        annotation_mask_vec(annotations)
        
        return masks
    
    load_luna16_seg_vec = np.vectorize(load_luna16_seg, otypes=[object])
    
    segs = load_luna16_seg_vec(subset_sids[:3])

    return segs



def load_nsclc_cts() -> np.ndarray:
    data_dir = f'{NSCLC_RAW_DATA_DIR}/NSCLC-Radiomics'
    
    all_roots = [f'{data_dir}/{elem}' for elem in os.listdir(data_dir)]
    all_roots = list(filter(lambda x: x[-7:] != 'LICENSE', all_roots))
    all_roots = np.array(all_roots, dtype=object)
    
    def load_nsclc_ct(fname: str) -> np.ndarray:
        ct_root = f'{fname}/{os.listdir(fname)[0]}'
        
        contents = [f'{ct_root}/{fldr}' for fldr in os.listdir(ct_root)]
        
        slides_fldrs = list(filter(lambda x: x[-8:-6] == 'NA', contents))
        slides_fldr = slides_fldrs[0] if len(os.listdir(slides_fldrs[0])) > 1 else slides_fldrs[1]
        
        def load(slide_fname):
            slide = dicom.dcmread(slide_fname).pixel_array

            return slide
        
        slide_fnames = [f'{slides_fldr}/{slide}' for slide in os.listdir(slides_fldr)]
        slide_fnames = np.array(slide_fnames, dtype=object)
        
        load_vec = np.vectorize(load, otypes=[object])
        
        ct = load_vec(slide_fnames)
        ct = np.vstack(ct)
        ct = ct.astype(float)
        ct = ct.reshape((-1, IMG_SIZE, IMG_SIZE))
        
        return ct

    loader = np.vectorize(load_nsclc_ct, otypes=[object])

    all_cts = loader(all_roots[:3])
    
    return all_cts



def load_nsclc_segs() -> np.ndarray:
    data_dir = f'{NSCLC_RAW_DATA_DIR}/NSCLC-Radiomics'
    
    all_roots = [f'{data_dir}/{elem}' for elem in os.listdir(data_dir)]
    all_roots = list(filter(lambda x: x[-7:] != 'LICENSE', all_roots))
    all_roots = np.array(all_roots, dtype=object)

    
    def load_nsclc_seg(fname):
        ct_root = f'{fname}/{os.listdir(fname)[0]}'
        
        contents = [f'{ct_root}/{fldr}' for fldr in os.listdir(ct_root)]
        
        seg_fldr = list(filter(lambda x: x[-8:-6] != 'NA', contents))[0]
        seg_fname = f'{seg_fldr}/{os.listdir(seg_fldr)[0]}' 
        
        ds = dicom.dcmread(seg_fname)
        
        raw_img = ds.pixel_array
        segment_items = ds.SegmentSequence
        
        count_segs = len(segment_items)
        count_slides = len(raw_img) // count_segs
        
        for idx, seg in enumerate(segment_items):
            if seg.SegmentDescription == 'GTV-1': break
        else:
            return np.zeros((count_slides, IMG_SIZE, IMG_SIZE))
        
        segment = raw_img[idx*count_slides : (idx+1)*count_slides]
        
        return segment

    loader = np.vectorize(load_nsclc_seg, otypes=[object])

    all_segs = loader(all_roots[:3])
    
    return all_segs


