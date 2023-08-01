# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:11:17 2023

@author: Gavin
"""

import torch, math, time, random

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from typing import List
from optim import CompositeLoss, compute_all_metrics

class SimpleBPOptimizer:
    
    def __init__(
        self,
        seed: int,
        model: nn.Module, 
        train: DataLoader, 
        valid: DataLoader,
        device: str='cpu'
    ) -> None:
        super().__init__()
        
        self.seed = seed
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


        
    def execute(
        self,
        epochs=100,
        start_epoch=0,
        lr=1e-5,
        cum_batch_size=32,
        valid_freq=10, 
        wbce_positive_frac=1,
        wbce_weight=1,
        dice_weight=100,
        perc_weight=1,
        verbose=True,
        checkpoint_callback=None,
        init_history = None
    ) -> List[dict]:
        
        if init_history is not None:
            history = []
        else:
            history = init_history

        start = time.time()

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = CompositeLoss(
            self.positive_weight, 
            wbce_positive_frac=wbce_positive_frac,
            wbce_weight=wbce_weight,
            dice_weight=dice_weight,
            perc_weight=perc_weight,
            device=self.device
        )
        
        print('#'*32)
        print('beginning BP training loop...')
        print('#'*32)
        
        for i in range(start_epoch, epochs):
            epoch = i + 1
            
            self.__reset_seed(self.seed + i)
            
            if verbose and i % 10 == 0:
                print(f'executing epoch {epoch}...', end='')
                
            history_record = {}
            
            train_num_slides = 0
            train_loss = 0
            
            model = self.model.train()
            
            metrics_dict = {}

            for batch in self.train_loader:
                xs, ys = batch

                optim.zero_grad()

                for ct, seg in zip(xs, ys):
                    num_chunks = math.ceil(ct.size(1) / cum_batch_size)

                    ct = ct.to(self.device)
                    seg = seg.to(self.device)

                    # split inputs into 'num_chunks' chunks for grad accumulation
                    ct_chunks = torch.chunk(ct, num_chunks, dim=1)
                    seg_chunks = torch.chunk(seg, num_chunks, dim=1)
                    
                    # ensure no chunks have only a single slide for BatchNorm layers
                    if len(ct_chunks) > 1 and ct_chunks[-1].size(1) == 1:
                        merged_ct_chunk = torch.cat((ct_chunks[-2], ct_chunks[-1]), dim=1)
                        ct_chunks = ct_chunks[:-2] + (merged_ct_chunk,)
                        
                        merged_seg_chunk = torch.cat((seg_chunks[-2], seg_chunks[-1]), dim=1)
                        seg_chunks = seg_chunks[:-2] + (merged_seg_chunk,)

                    for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                        ct_chunk = ct_chunk.unsqueeze(dim=0).float()
                        seg_chunk = seg_chunk.unsqueeze(dim=0).float()

                        pred_chunk = model(ct_chunk)

                        chunk_loss = criterion(pred_chunk, seg_chunk)
                        chunk_loss.backward() # accumulate gradients
                        
                        batch_size = ct_chunk.shape[1]
                        
                        train_num_slides += batch_size
                        train_loss += chunk_loss.item()
                        
                        metric_scores = compute_all_metrics(pred_chunk, seg_chunk)
                        
                        for name, score in metric_scores.items():
                            if name not in metrics_dict.keys():
                                metrics_dict[name] = score * batch_size
                            else:
                                metrics_dict[name] += score * batch_size


                optim.step() # update model with accumulated gradients
            
            history_record['train_loss'] = train_loss
            history_record['train_norm_loss'] = train_loss / train_num_slides
            
            wavg_metrics = {
                f'train_{name}': w_score / train_num_slides for name, w_score in metrics_dict.items()
            }
            
            history_record.update(wavg_metrics)

            if i % valid_freq == 0 or epoch == epochs:
                valid_num_slides = 0
                valid_loss = 0
    
                model = self.model.eval()
                
                metrics_dict = {}
    
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
                            
                            merged_seg_chunk = torch.cat((seg_chunks[-2], seg_chunks[-1]), dim=1)
                            seg_chunks = seg_chunks[:-2] + (merged_seg_chunk,)
                            
                        for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                            ct_chunk = ct_chunk.unsqueeze(dim=0).float()
                            seg_chunk = seg_chunk.unsqueeze(dim=0).float()
    
                            pred_chunk = model(ct_chunk)
    
                            chunk_loss = criterion(pred_chunk, seg_chunk)
                            
                            batch_size = ct_chunk.shape[1]
                                
                            valid_num_slides += batch_size
                            valid_loss += chunk_loss.item()
                            
                            metric_scores = compute_all_metrics(pred_chunk, seg_chunk)
                            
                            for name, score in metric_scores.items():
                                if name not in metrics_dict.keys():
                                    metrics_dict[name] = score * batch_size
                                else:
                                    metrics_dict[name] += score * batch_size

                history_record['valid_loss'] = valid_loss
                history_record['valid_norm_loss'] = valid_loss / valid_num_slides
                
                wavg_metrics = {
                    f'valid_{name}': w_score / valid_num_slides for name, w_score in metrics_dict.items()
                }
                
                history_record.update(wavg_metrics)

            history.append(history_record)
            
            checkpoint_callback(history, epoch)

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
    
    
    
    def __reset_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)