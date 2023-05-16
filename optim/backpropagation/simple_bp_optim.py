# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:11:17 2023

@author: Gavin
"""

import torch, math, time

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from typing import List
from optim import WeightedBCELoss, BinaryDiceLoss

class SimpleBPOptimizer:
    
    def __init__(self,
        model: nn.Module, 
        train: DataLoader, 
        valid: DataLoader,
        device: str='cpu'
    ) -> None:
        super().__init__()
        
        self.model = model
        self.train_loader = train
        self.valid_loader = valid
        self.device = device
        self.positive_weight = self.__compute_positive_weight()

        self.train_slides = sum([sum([x.size()[1] for x in x_batch]) for x_batch, _ in self.train_loader])
        self.valid_slides = sum([sum([x.size()[1] for x in x_batch]) for x_batch, _ in self.valid_loader])



    def __compute_positive_weight(self):
        positive_voxels = 0
        negative_voxels = 0

        for batch in self.train_loader:
            _, ys = batch
            
            for y in ys:
                y = y.cpu().detach().numpy()
    
                positive = y.sum()
                total = np.array(y.shape).prod()
                negative = total - positive
    
                positive_voxels += positive
                negative_voxels += negative

        return negative_voxels / positive_voxels


        
    def execute(self, epochs=100, lr=1e-5, cum_batch_size=32, valid_freq=10, verbose=True) -> List[dict]:
        history = []

        start = time.time()

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = WeightedBCELoss(self.positive_weight)

        compute_dice = BinaryDiceLoss()
        
        print('#'*32)
        print('beginning BP training loop...')
        print('#'*32)
        
        for i in range(epochs):
            epoch = i + 1
            
            if verbose and i % 10 == 0:
                print(f'executing epoch {epoch}...', end='')
                
            history_record = {}
            
            train_num_slides = 0
            train_loss = 0
            train_correct_pixels = 0
            train_total_pixels = 0
            train_tp_pixels = 0
            train_gt_pixels = 0
            train_fp_pixels = 0
            train_fn_pixels = 0
            train_cum_dice = 0
            
            model = self.model.train()

            for batch in self.train_loader:
                xs, ys = batch

                optim.zero_grad()

                for ct, seg in zip(xs, ys):
                    num_chunks = math.ceil(ct.size(1) / cum_batch_size)

                    ct = ct.to(self.device)
                    seg = seg.to(self.device)

                    ct_chunks = torch.chunk(ct, num_chunks, dim=1)
                    seg_chunks = torch.chunk(seg, num_chunks, dim=1)
                    
                    if len(ct_chunks) > 1 and ct_chunks[-1].size(1) == 1:
                        merged_ct_chunk = torch.cat((ct_chunks[-2], ct_chunks[-1]), dim=1)
                        ct_chunks = ct_chunks[:-2] + (merged_ct_chunk,)
                    
                    if len(seg_chunks) > 1 and seg_chunks[-1].size(1) == 1:
                        merged_seg_chunk = torch.cat((seg_chunks[-2], seg_chunks[-1]), dim=1)
                        seg_chunks = seg_chunks[:-2] + (merged_seg_chunk,)

                    for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                        ct_chunk = ct_chunk.unsqueeze(dim=0).float()
                        seg_chunk = seg_chunk.unsqueeze(dim=0).float()

                        pred_chunk = model(ct_chunk)
                        pred_mask = (pred_chunk > 0.5).bool()

                        train_correct_pixels += (pred_mask == seg_chunk).sum().item()
                        train_total = np.array(ct_chunk.size()).prod()
                        train_total_pixels += train_total

                        train_tp_mask = torch.logical_and(pred_mask, seg_chunk.to(torch.bool))

                        train_tp_pixels += train_tp_mask.reshape(train_tp_mask.size(0), -1).sum(dim=1).item()
                        train_gt = seg_chunk.reshape(seg_chunk.size(0), -1).sum(dim=1).item()
                        train_gt_pixels += train_gt

                        train_fp_mask = torch.logical_or(pred_mask, ~seg_chunk.to(torch.bool))

                        train_fp_pixels += train_fp_mask.reshape(train_fp_mask.size(0), -1).sum(dim=1).item()
                        train_fn_pixels += train_total - train_gt

                        chunk_loss = criterion(pred_chunk, seg_chunk)
                        chunk_loss.backward()
                        
                        train_num_slides += len(ct_chunk)
                        train_loss += chunk_loss.item()

                        dice = compute_dice(pred_chunk, seg_chunk).item()
                        norm_dice = dice * (seg_chunk.size()[1] / self.train_slides)

                        train_cum_dice += norm_dice

                optim.step()
            
            history_record['train_loss'] = train_loss
            history_record['train_norm_loss'] = train_loss / train_num_slides
            history_record['train_acc'] = train_correct_pixels / train_total_pixels
            history_record['train_tpr'] = train_tp_pixels / train_gt_pixels
            history_record['train_fpr'] = train_fp_pixels / train_fn_pixels
            history_record['train_dice'] = train_cum_dice

            if i % valid_freq == 0:
                valid_num_slides = 0
                valid_loss = 0
                valid_correct_pixels = 0
                valid_total_pixels = 0
                valid_tp_pixels = 0
                valid_gt_pixels = 0
                valid_fp_pixels = 0
                valid_fn_pixels = 0
                valid_cum_dice = 0
    
                model = self.model.eval()
    
                for batch in self.valid_loader:
                    xs, ys = batch
                        
                    for ct, seg in zip(xs, ys):
                        num_chunks = math.ceil(ct.size(1) / cum_batch_size)
                            
                        ct = ct.to(self.device)
                        seg = seg.to(self.device)
                        
                        ct_chunks = torch.chunk(ct, num_chunks, dim=1)
                        seg_chunks = torch.chunk(seg, num_chunks, dim=1)
                        
                        if len(ct_chunks) > 1 and ct_chunks[-1].size(1) == 1:
                            merged_ct_chunk = torch.cat((ct_chunks[-2], ct_chunks[-1]), dim=1)
                            ct_chunks = ct_chunks[:-2] + (merged_ct_chunk,)
                        
                        if len(seg_chunks) > 1 and seg_chunks[-1].size(1) == 1:
                            merged_seg_chunk = torch.cat((seg_chunks[-2], seg_chunks[-1]), dim=1)
                            seg_chunks = seg_chunks[:-2] + (merged_seg_chunk,)
                            
                        for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                            ct_chunk = ct_chunk.unsqueeze(dim=0).float()
                            seg_chunk = seg_chunk.unsqueeze(dim=0).float()
    
                            pred_chunk = model(ct_chunk)
                            pred_mask = (pred_chunk > 0.5).bool()
    
                            valid_correct_pixels += (pred_mask == seg_chunk).sum().item()
                            valid_total = np.array(ct_chunk.size()).prod()
                            valid_total_pixels += valid_total
    
                            valid_tp_mask = torch.logical_and(pred_mask, seg_chunk.to(torch.bool))
    
                            valid_tp_pixels += valid_tp_mask.reshape(valid_tp_mask.size(0), -1).sum(dim=1).item()
                            valid_gt = seg_chunk.reshape(seg_chunk.size(0), -1).sum(dim=1).item()
                            valid_gt_pixels += valid_gt
    
                            valid_fp_mask = torch.logical_or(pred_mask, ~seg_chunk.to(torch.bool))
    
                            valid_fp_pixels += valid_fp_mask.reshape(valid_fp_mask.size(0), -1).sum(dim=1).item()
                            valid_fn_pixels += valid_total - valid_gt
    
                            chunk_loss = criterion(pred_chunk, seg_chunk)
                                
                            valid_num_slides += len(ct_chunk)
                            valid_loss += chunk_loss.item()

                            dice = compute_dice(pred_chunk, seg_chunk).item()
                            norm_dice = dice * (seg_chunk.size()[1] / self.valid_slides)

                            valid_cum_dice += norm_dice
                                
                history_record['valid_loss'] = valid_loss
                history_record['valid_norm_loss'] = valid_loss / valid_num_slides
                history_record['valid_acc'] = valid_correct_pixels / valid_total_pixels
                history_record['valid_tpr'] = valid_tp_pixels / valid_gt_pixels
                history_record['valid_fpr'] = valid_fp_pixels / valid_fn_pixels
                history_record['valid_dice'] = valid_cum_dice

            history.append(history_record)

            if verbose and i % 10 == 0 and epoch != epochs:
                print('done')
                print(f'epoch {epoch} training statistics:')
                print('\n'.join([f'->{key} = {value:.4f}' for key, value in history_record.items()]))
                print('-'*32)
            
        print('#'*32)
        print('finished BP training loop!')
        print('final training statistics:')
        print('\n'.join([f'->{key} = {value:.4f}' for key, value in history[-1].items()]))
        print('#'*32)

        end = time.time()

        print(f'total elapsed time: {end-start}s')

        return history
            