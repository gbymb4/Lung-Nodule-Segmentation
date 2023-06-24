# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:51:47 2023

@author: Gavin
"""

import imageio, os, time, json, torch, yaml

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from torch import nn
from typing import List
from pconfig import NSCLC_PREPROCESSED_DATA_DIR, LUNA16_PREPROCESSED_DATA_DIR, OUT_DIR

def plot_and_save_gif(
    images: np.ndarray,
    fname: str,
    titles: List[str]=None,
    verbose: bool=False,
    fps: int=10
) -> None:
    
    fig_handles = []
    
    for img_tup in images:
        fig, axs = plt.subplots(1, images.shape[1])
        
        for i, (img, ax) in enumerate(zip(img_tup, axs)):
            if titles is not None: 
                ax.set_title(titles[i], fontsize=14)
                
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        fig.tight_layout()
        
        fig_handles.append(fig)

    with imageio.get_writer(fname, mode='I', fps=fps) as writer:
        for i, fig in enumerate(fig_handles):
            
            if verbose and i % 10 == 0: print(f'Rendering frame {i + 1}...')
            
            fig.canvas.draw()
            
            img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            writer.append_data(img_np)

    for fig in fig_handles:
        plt.close(fig)
        


def plot_and_save_metric(train, valid, metric, fname):
    fig, ax = plt.subplots(figsize=(8, 6))

    epochs = list(range(1, len(train) + 1))

    ax.plot(epochs, train, label='Train', alpha=0.7)
    ax.plot(epochs, valid, label='Validation', alpha=0.7)   
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='major', length=4)
    ax.tick_params(which='minor', length=2, color='r')
    
    ax.legend()
    ax.grid(axis='y', c='white')
    
    ax.set_facecolor('whitesmoke')
    
    metric_name = (' '.join(metric.split('_'))).title()
    
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18)
    
    plt.savefig(fname)
    plt.show()
    
    
    
def plot_and_save_slide(slide, fname):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.axis('off')
    ax.imshow(slide, cmap='gray')
    
    plt.savefig(fname)
    plt.show()
    


def save_instance(
    dataset: str,
    partition: str,
    subset: int,
    idx: int,
    xs: np.ndarray,
    ys: np.ndarray
) -> None:

    if partition == 'train':
    
        if dataset.lower() == 'nsclc':
            if not os.path.isdir(f'{NSCLC_PREPROCESSED_DATA_DIR}/train'): 
                os.mkdir(f'{NSCLC_PREPROCESSED_DATA_DIR}/train')
                
            save_dir = f'{NSCLC_PREPROCESSED_DATA_DIR}/train/subset{subset}'
            
        elif dataset.lower() == 'luna16':
            if not os.path.isdir(f'{LUNA16_PREPROCESSED_DATA_DIR}/train'): 
                os.mkdir(f'{LUNA16_PREPROCESSED_DATA_DIR}/train')
                
            save_dir = f'{LUNA16_PREPROCESSED_DATA_DIR}/train/subset{subset}'
        else:
            raise ValueError(f'dataset "{dataset}" does not exist')  
    
    elif partition == 'test':
        
        if dataset.lower() == 'nsclc':
            save_dir = f'{NSCLC_PREPROCESSED_DATA_DIR}/test'
        elif dataset.lower() == 'luna16':
            save_dir = f'{LUNA16_PREPROCESSED_DATA_DIR}/test'
        else:
            raise ValueError(f'dataset "{dataset}" does not exist')
    
    else:
        raise ValueError(f'partition "{partition}" does not exist')
    
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    
    np.savez_compressed(f'{save_dir}/scan_{idx}.npz', x=xs, y=ys)



def save_history_dict_and_model(
    dataset: str,
    model: nn.Module,
    id: int,
    config: dict,
    train_idx: int,
    history: dict
) -> None:
    dataset = dataset.lower()
    model_name = type(model).__name__

    if not os.path.isdir(OUT_DIR): os.mkdir(OUT_DIR)

    save_root_dir = f'{OUT_DIR}/{dataset}'
    if not os.path.isdir(save_root_dir): os.mkdir(save_root_dir)

    save_parent_dir = f'{save_root_dir}/{model_name}'
    if not os.path.isdir(save_parent_dir): os.mkdir(save_parent_dir)

    save_dir = f'{save_parent_dir}/{train_idx}_{id}'
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    with open(f'{save_dir}/history.json', 'w') as file:
        json.dump(history, file)

    with open(f'{save_dir}/config.yaml', 'w') as file:
        yaml.safe_dump(config, file)
        
    for param in model.parameters():
        param.requires_grad = True

    torch.save(model.state_dict(), f'{save_dir}/model')