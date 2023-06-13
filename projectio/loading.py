# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:51:42 2023

@author: Gavin
"""

import os, torch

import numpy as np
import skimage.io as io
import pydicom as dicom
import pylidc as pl
import matplotlib.path as mpath

from pconfig import (
    LUNA16_RAW_DATA_DIR, 
    NSCLC_RAW_DATA_DIR, 
    LUNA16_PREPROCESSED_DATA_DIR, 
    NSCLC_PREPROCESSED_DATA_DIR,
    IMG_SIZE
)
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Union, Iterable, Callable, Tuple, List
from multiprocessing import Pool
from pylidc.utils import consensus
from skimage.measure import find_contours

def luna16_ct_fnames(subset: int, load_limit: Union[int, None]=None) -> np.ndarray:
    assert load_limit is None or load_limit > 0
    
    data_dir = f'{LUNA16_RAW_DATA_DIR}/subset{subset}'
    
    all_headers = list(filter(lambda x: x[-4:] == '.mhd', os.listdir(data_dir)))
    all_headers = np.array([f'{data_dir}/{elem}' for elem in all_headers], dtype=object)
    
    if load_limit is not None:
        all_headers = all_headers[:load_limit]
        
    return all_headers
        
        

def load_luna16_cts(subset: int, load_limit: int=None) -> np.ndarray:
    all_headers = luna16_ct_fnames(subset, load_limit)

    loader = np.vectorize(load_luna16_ct, otypes=[object])
    all_cts = loader(all_headers)
    
    return all_cts



def load_luna16_ct(fname: str) -> np.ndarray:
    ct = io.imread(fname, plugin='simpleitk')  
    
    return ct



def luna16_seg_subset_sids(subset: int, load_limit: Union[int, None]=None) -> np.ndarray:
    assert load_limit is None or load_limit > 0
    
    data_dir = f'{LUNA16_RAW_DATA_DIR}/subset{subset}'
    
    all_headers = list(filter(lambda x: x[-4:] == '.mhd', os.listdir(data_dir)))
    subset_sids = np.array([header[:-4] for header in all_headers], dtype=object)
    
    return subset_sids



def luna16_seg_sids(load_limit: Union[int, None]=None) -> np.ndarray:
    assert load_limit is None or load_limit > 0
    
    data_dir = f'{LUNA16_RAW_DATA_DIR}/lungs'
    
    all_headers = list(filter(lambda x: x[-4:] == '.mhd', os.listdir(data_dir)))
    sids = np.array([header[:-4] for header in all_headers], dtype=object)
    
    return sids



def load_luna16_segs(subset: int, load_limit: Union[int, None]=None) -> np.ndarray:
    subset_sids = luna16_seg_subset_sids()
    
    load_luna16_seg_vec = np.vectorize(load_luna16_seg, otypes=[object])
    
    if load_limit is not None:
        subset_sids = subset_sids[:load_limit]
    
    segs = load_luna16_seg_vec(subset_sids)

    return segs



def load_luna16_censensus_contours(scan: pl.Scan) -> List[List[Tuple[int, np.ndarray]]]:
    clusters = scan.cluster_annotations()
    
    consensus_contours = []
    
    for j, cluster in enumerate(clusters):
        cmask, cbbox, masks = consensus(cluster)
        
        start = cbbox[2].start
        stop = cbbox[2].stop
        
        bbox_origin_x = cbbox[1].start
        bbox_origin_y = cbbox[0].start
        
        contours = []
        
        for i in range(stop - start - 1):
            nodule_contours = find_contours(cmask[:, :, i].astype(float), 0.5)
            
            for contour_region in nodule_contours:
                contour_region[:, [0, 1]] = contour_region[:, [1, 0]]
                
                contour_region[:, 0] += bbox_origin_x
                contour_region[:, 1] += bbox_origin_y
            
                contours.append((start + i, contour_region))
            
        consensus_contours.append(contours)
        
    return consensus_contours
    



def load_luna16_seg(sid: str) -> np.ndarray:
    scan = [scan for scan in pl.query(pl.Scan) if scan.series_instance_uid == sid][0]
    
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



def nsclc_ct_fnames(load_limit: Union[int, None]=None) -> np.ndarray:
    assert load_limit is None or load_limit > 0
    
    data_dir = f'{NSCLC_RAW_DATA_DIR}/NSCLC-Radiomics'
    
    all_roots = [f'{data_dir}/{elem}' for elem in os.listdir(data_dir)]
    all_roots = list(filter(lambda x: x[-7:] != 'LICENSE', all_roots))
    all_roots = np.array(all_roots, dtype=object)
    
    return all_roots



def load_nsclc_cts(load_limit: Union[int, None]=None) -> np.ndarray:
    all_roots = nsclc_ct_fnames(load_limit)

    loader = np.vectorize(load_nsclc_ct, otypes=[object])
    
    if load_limit is not None:
        all_roots = all_roots[:load_limit]

    all_cts = loader(all_roots)
    
    return all_cts



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



def nsclc_seg_fnames(load_limit: Union[int, None]=None) -> np.ndarray:
    assert load_limit is None or load_limit > 0
    
    data_dir = f'{NSCLC_RAW_DATA_DIR}/NSCLC-Radiomics'
    
    all_roots = [f'{data_dir}/{elem}' for elem in os.listdir(data_dir)]
    all_roots = list(filter(lambda x: x[-7:] != 'LICENSE', all_roots))
    all_roots = np.array(all_roots, dtype=object)

    return all_roots
    
    

def load_nsclc_segs(load_limit: Union[int, None]=None) -> np.ndarray:
    all_roots = nsclc_seg_fnames(load_limit)

    loader = np.vectorize(load_nsclc_seg, otypes=[object])

    if load_limit is not None:
        all_roots = all_roots[:load_limit]

    all_segs = loader(all_roots)
    
    return all_segs



def load_nsclc_seg(fname: str) -> np.ndarray:
    ct_root = f'{fname}/{os.listdir(fname)[0]}'
    
    contents = [f'{ct_root}/{fldr}' for fldr in os.listdir(ct_root)]
    
    fldrs = list(filter(lambda x: x[-8:-6] != 'NA', contents))
    
    if len(fldrs) == 0:
        return None
    
    seg_fldr = fldrs[0]
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



class LNSegDataset(Dataset):
    
    def __init__(
        self,
        dataset: str,
        partition: str,
        subset: int,
        load_ct_dims: Union[List[int], None]=None,
        load_limit: Union[int, None]=None, 
        device: str='cpu',
        transforms: Union[Iterable[Callable], None]=None,
        transform_kwargs: Union[Iterable[dict], None]=None
    ) -> None:
        super().__init__()
        
        assert load_limit is None or load_limit > 0
        
        if partition == 'train':
        
            if dataset.lower() == 'luna16':
                data_dir = f'{LUNA16_PREPROCESSED_DATA_DIR}/train/subset{subset}'
            elif dataset.lower() == 'nsclc':
                data_dir = f'{NSCLC_PREPROCESSED_DATA_DIR}/train/subset{subset}'
            else:
                raise ValueError(f"invalid value for arg 'dataset': {dataset}")
                
        elif partition == 'test':
            
            if dataset.lower() == 'luna16':
                data_dir = f'{LUNA16_PREPROCESSED_DATA_DIR}/test'
            elif dataset.lower() == 'nsclc':
                data_dir = f'{NSCLC_PREPROCESSED_DATA_DIR}/test'
            else:
                raise ValueError(f"invalid value for arg 'dataset': {dataset}")
            
        else:
            raise ValueError(f"invalid value for arg 'partition': {partition}")
                
        scan_fnames = [f'{data_dir}/{scan_fname}' for scan_fname in os.listdir(data_dir)]
     
        self.load_all(
            scan_fnames, 
            load_ct_dims, 
            transforms, 
            transform_kwargs, 
            device
        )   
     
     
     
    def __len__(self) -> None:
         return len(self.xs)
     
        
     
    def __getitem__(self, idx: Union[Iterable, int]) -> None:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        xs = self.xs[idx]
        ys = self.ys[idx]
        
        return xs, ys
    
    
    
    def load_instance(
        self,
        scan_fname: str,
        load_ct_dims: Iterable[int],
        transforms: Iterable[Callable],
        transform_kwargs: Iterable[dict],
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        data = np.load(scan_fname)
        
        x, y = data['x'], data['y']

        x = x.swapaxes(1, 3)
        x = x.swapaxes(0, 1)

        y = y[:, :, :, np.newaxis].swapaxes(1, 3)
        y = y.swapaxes(0, 1)

        if load_ct_dims is not None:
            x = x[np.array(load_ct_dims), :, :, :]

        if transforms is not None:
            for transform, kwargs in zip(transforms, transform_kwargs):
                x, y = transform(x, y, **kwargs)
        
        x = torch.from_numpy(x).type(torch.float16)
        x = x.to(device)
        
        y = torch.from_numpy(y).type(torch.int8)
        y = y.to(device)
        
        return x, y
    
    
    
    def load_all(self, scan_fnames: Iterable[str], *args) -> None:
        self.xs, self.ys = [], []
        
        for scan_fname in scan_fnames:
            x, y = self.load_instance(scan_fname, *args)
            
            self.xs.append(x)
            self.ys.append(y)



class LNSegDatasetNodules(LNSegDataset):
            
    def load_instance(
        self,
        scan_fname: str,
        load_ct_dims: Iterable[int],
        transforms: Iterable[Callable],
        transform_kwargs: Iterable[dict],
        device: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        data = np.load(scan_fname)
        
        x, y = data['x'], data['y']

        x = x.swapaxes(1, 3)
        x = x.swapaxes(0, 1)

        y = y[:, :, :, np.newaxis].swapaxes(1, 3)
        y = y.swapaxes(0, 1)

        if load_ct_dims is not None:
            x = x[np.array(load_ct_dims), :, :, :]

        filter_idxs = y.reshape(y.shape[1], -1).sum(axis=1) > 0
        dilated_idxs = np.convolve(filter_idxs, np.array([1, 1]), mode='same')
        filter_idxs[dilated_idxs > 0] = 1

        filter_idx_regions = self.__split_connected_regions(filter_idxs)

        x_splits, y_splits = [], []

        for i, region_idxs in enumerate(filter_idx_regions):
            x_splits.append(x[:, np.array(region_idxs), :, :])
            y_splits.append(y[:, np.array(region_idxs), :, :])

        for i, (x_split, y_split) in enumerate(zip(x_splits, y_splits)):
            if transforms is not None:
                for transform, kwargs in zip(transforms, transform_kwargs):
                    x_split, y_split = transform(x_split, y_split, **kwargs)

            x_split = torch.from_numpy(x_split).type(torch.float16)
            x_split = x_split.to(device)

            y_split = torch.from_numpy(y_split).type(torch.int8)
            y_split = y_split.to(device)

            x_splits[i] = x_split
            y_splits[i] = y_split

        return x_splits, y_splits

        

    def load_all(self, scan_fnames: Iterable[str], *args) -> None:
        self.xs, self.ys = [], []
        
        for scan_fname in scan_fnames:
            x_splits, y_splits = self.load_instance(scan_fname, *args)

            for x_split, y_split in zip(x_splits, y_splits):
                self.xs.append(x_split)
                self.ys.append(y_split)



    def __split_connected_regions(self, arr):
        indices = np.where(arr)[0]
        regions = []
        region = []
        prev_index = None

        for index in indices:
            if prev_index is None or index == prev_index + 1:
                region.append(index)
            else:
                regions.append(region)
                region = [index]
            prev_index = index

        if region: regions.append(region)

        return regions
    


def prepare_dataset(inputs):
    return inputs[0][0](*inputs[0][1:], **inputs[1])



def prepare_datasets(dataset, dataset_type, partition, num_workers=1, **kwargs):
    assert num_workers > 0

    if dataset.lower() == 'luna16':
        data_dir = LUNA16_PREPROCESSED_DATA_DIR
    elif dataset.lower() == 'nsclc':
        data_dir = NSCLC_PREPROCESSED_DATA_DIR
    else:
        raise ValueError(f"invalid value for arg 'dataset': {dataset}")
        
    subsets_count = len([subset_dir for subset_dir in os.listdir(data_dir) if subset_dir[:6] == 'subset'])

    if num_workers == 1:
        datasets = [prepare_dataset(dataset_type, dataset, partition, i, **kwargs) for i in range(subsets_count)]
    else:
        all_args = [(dataset_type, dataset, partition, i) for i in range(subsets_count)]
        all_kwargs = [kwargs for _ in range(subsets_count)]
        pool_arguments = list(zip(all_args, all_kwargs))

        with Pool(num_workers) as p:
            datasets = p.map(prepare_dataset, pool_arguments)

    return datasets


    
def prepare_dataloaders(datasets, train_idx, **kwargs):
    all_idxs = np.arange(0, len(datasets))
    
    valid_idxs = all_idxs[all_idxs != train_idx]
    valid_datasets = [datasets[idx] for idx in valid_idxs]
    
    train = DataLoader(datasets[train_idx], collate_fn=__collate_fn, **kwargs)
    valid = DataLoader(ConcatDataset(valid_datasets), collate_fn=__collate_fn, **kwargs)
    
    return train, valid


def __collate_fn(batch):
    return list(zip(*batch))
