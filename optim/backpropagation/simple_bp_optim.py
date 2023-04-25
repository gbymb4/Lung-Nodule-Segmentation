# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:11:17 2023

@author: Gavin
"""

import torch

from torch import nn
from torch.utils.data import DataLoader
from typing import List

class SimpleBPOptimizer:
    
    def __init__(self,
        model: nn.Module, 
        train: DataLoader, 
        valid: DataLoader,
        device: str='cpu'
    ) -> None:
        super().__init__()
        
        self.model = model.half()
        self.train_loader = train
        self.valid_loader = valid
        self.device = device
        
        
        
    def execute(self, epochs=100, lr=1e-5, cum_batch_size=32, verbose=True) -> List[dict]:
        history = []
        
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        print('#'*32)
        print('beginning BP training loop...')
        print('#'*32)
        
        for i in range(epochs):
            epoch = i + 1
            
            if verbose and i % 10 == 0:
                print(f'executing epoch {epoch}...')
                
            history_record = {}
            
            train_num_slides = 0
            train_loss = 0
            train_correct_pixels = 0
            train_total_pixels = 0
            train_tp_pixels = 0
            train_gt_pixels = 0
            train_fp_pixels = 0
            train_fn_pixels = 0
            
            model = self.model.train()
            
            for batch in self.train_loader:
                xs, ys = batch
                
                optim.zero_grad()
                
                for ct, seg in zip(xs, ys):
                    num_chunks = len(xs) // cum_batch_size
                    
                    ct = ct.to(self.device)
                    seg = seg.to(self.device)
                    
                    ct_chunks = torch.chunk(ct, num_chunks, dim=1)
                    seg_chunks = torch.chunk(seg, num_chunks, dim=1)
                    
                    for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                        pred_chunk = model(ct_chunk)
                        pred_mask = (pred_chunk > 0.5).bool()

                        train_correct_pixels += (pred_mask == seg_chunk).sum().item()
                        train_total = ct_chunk.shape.cpu().detach().numpy().prod()
                        train_total_pixels += train_total

                        train_tp_mask = torch.logical_and(pred_mask, seg_chunk)

                        train_tp_pixels += train_tp_mask.view(train_tp_mask.size(0), -1).sum(dim=1).item()
                        train_gt = seg_chunk.view(seg_chunk.size(0), -1).sum(dim=1).item()
                        train_gt_pixels += train_gt

                        train_fp_mask = torch.logical_or(pred_mask, ~seg_chunk)

                        train_fp_pixels += train_fp_mask.view(train_fp_mask.size(0), -1).sum(dim=1).item()
                        train_fn_pixels += train_total - train_gt

                        chunk_loss = criterion(seg_chunk, pred_chunk)
                        chunk_loss.backward()
                        
                        train_num_slides += len(ct_chunk)
                        train_loss += chunk_loss.item()
                        
                optim.step()
            
            history_record['train_loss'] = train_loss
            history_record['train_norm_loss'] = train_loss / train_num_slides
            history_record['train_acc'] = train_correct_pixels / train_total_pixels
            history_record['train_tpr'] = train_tp_pixels / train_gt_pixels
            history_record['train_fpr'] = train_fp_pixels / train_fn_pixels

            valid_num_slides = 0
            valid_loss = 0
            valid_correct_pixels = 0
            valid_total_pixels = 0
            valid_tp_pixels = 0
            valid_gt_pixels = 0
            valid_fp_pixels = 0
            valid_fn_pixels = 0
            
            model = self.model.eval()
            
            with torch.no_grad():
                for batch in self.valid_loader:
                    xs, ys = batch
                    
                    optim.zero_grad()
                    
                    for ct, seg in zip(xs, ys):
                        num_chunks = len(xs) // cum_batch_size
                        
                        ct = ct.to(self.device)
                        seg = seg.to(self.device)
                        
                        ct_chunks = torch.chunk(ct, num_chunks, dim=1)
                        seg_chunks = torch.chunk(seg, num_chunks, dim=1)
                        
                        for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                            pred_chunk = model(ct_chunk)
                            pred_mask = (pred_chunk > 0.5).bool()

                            valid_correct_pixels += (pred_mask == seg_chunk).sum().item()
                            valid_total = ct_chunk.shape.cpu().detach().numpy().prod()
                            valid_total_pixels += valid_total

                            valid_tp_mask = torch.logical_and(pred_mask, seg_chunk)

                            valid_tp_pixels += valid_tp_mask.view(valid_tp_mask.size(0), -1).sum(dim=1).item()
                            valid_gt = seg_chunk.view(seg_chunk.size(0), -1).sum(dim=1).item()
                            valid_gt_pixels += valid_gt

                            valid_fp_mask = torch.logical_or(pred_mask, ~seg_chunk)

                            valid_fp_pixels += valid_fp_mask.view(valid_fp_mask.size(0), -1).sum(dim=1).item()
                            valid_fn_pixels += valid_total - valid_gt

                            chunk_loss = criterion(seg_chunk, pred_chunk)
                            
                            valid_num_slides += len(ct_chunk)
                            valid_loss += chunk_loss.item()
                            
            history_record['valid_loss'] = valid_loss
            history_record['valid_norm_loss'] = valid_loss / valid_num_slides
            history_record['valid_acc'] = valid_correct_pixels / valid_total_pixels
            history_record['valid_tpr'] = valid_tp_pixels / valid_gt_pixels
            history_record['valid_fpr'] = valid_fp_pixels / valid_fn_pixels
            
            history.append(history_record)
            
            if verbose and i % 10 == 0 and epoch != epochs:
                print(f'epoch {epoch} training statistics:')
                print('\n'.join([f'->{key} = {value:.4f}' for key, value in history_record]))
                print('-'*32)
            
        print('#'*32)
        print('finished BP training loop!')
        print('final training statistics:')
        print('\n'.join([f'->{key} = {value:.4f}' for key, value in history[-1]]))
        print('#'*32)
            
        return history
            