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
            
            model = self.model.train()
            
            for batch in self.train_loader:
                xs, ys = batch
                
                optim.zero_grad()
                
                for ct, seg in zip(xs, ys):
                    num_chunks = len(xs) // cum_batch_size
                    
                    ct = ct.to(self.device)
                    seg = seg.to(self.device)
                    
                    ct_chunks = torch.chunk(ct, num_chunks, dim=1)
                    seg_chunks = torch.chunk(ct, num_chunks, dim=1)
                    
                    for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                        pred_chunk = model(ct_chunk)
                        
                        chunk_loss = criterion(seg_chunk, pred_chunk)
                        chunk_loss.backward()
                        
                        train_num_slides += len(ct_chunk)
                        train_loss += chunk_loss.item()
                        
                optim.step()
            
            history_record['train_loss'] = train_loss
            history_record['train_norm_loss'] = train_loss / train_num_slides
            
            valid_num_slides = 0
            valid_loss = 0
            
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
                        seg_chunks = torch.chunk(ct, num_chunks, dim=1)
                        
                        for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                            pred_chunk = model(ct_chunk)
                            
                            chunk_loss = criterion(seg_chunk, pred_chunk)
                            
                            valid_num_slides += len(ct_chunk)
                            valid_loss += chunk_loss.item()
                            
            history_record['valid_loss'] = valid_loss
            history_record['valid_norm_loss'] = valid_loss / valid_num_slides
            
            history.append(history_record)
            
            if verbose and i % 10 == 0 and epoch != epochs:
                print('-'*32)
            
        print('#'*32)
        print('finished BP training loop!')
        print('final training statistics:')
        print('\n'.join([f'->{key} = {value:.4f}' for key, value in history[-1]]))
        print('#'*32)
            
        return history
            