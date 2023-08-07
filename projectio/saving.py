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
from copy import deepcopy
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
    root_id: int,
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
    
    save_model_dir = f'{save_root_dir}/{model_name}'
    if not os.path.isdir(save_model_dir): os.mkdir(save_model_dir)

    save_root_id_dir = f'{save_model_dir}/{root_id}'
    if not os.path.isdir(save_root_id_dir): os.mkdir(save_root_id_dir)

    save_dir = f'{save_root_id_dir}/{train_idx}_{id}'
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    
    checkpoint_dir = f'{save_dir}/model_checkpoints'
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)

    with open(f'{save_dir}/history.json', 'w') as file:
        json.dump(history, file)

    with open(f'{save_dir}/config.yaml', 'w') as file:
        yaml.safe_dump(config, file)
        
    for param in deepcopy(model).parameters():
        param.requires_grad = True

    epoch = len(history)

    torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.pt')
    
    
    
def dump_preds_gif_and_metrics_plots(
    dataset, 
    model, 
    history,
    valid_loader, 
    device, 
    train_idx, 
    id,
    root_id,
    config_dict
):
    model = model.eval()

    valid_instance_example = next(iter(valid_loader))
    x, y = valid_instance_example
    x, y = x[0], y[0]
    
    x = x.unsqueeze(dim=0).float().to(device)
    y = y.unsqueeze(dim=0).float().to(device)

    pred_y_raw = model(x)
    pred_y = pred_y_raw > 0.5

    print('-'*32)
    print('saving training history and model...')

    save_history_dict_and_model(
        dataset,
        model, 
        root_id,
        id, 
        config_dict, 
        train_idx, 
        history
    )

    gif_name = f'{OUT_DIR}/{dataset.lower()}/{type(model).__name__}/{root_id}/{train_idx}_{id}/example_preds.gif'

    x = x[0].detach().float().cpu().numpy()
    x = x.swapaxes(0, 1)
    x = x.swapaxes(1, 3)

    y = y[0].detach().float().cpu().numpy()
    y = y.swapaxes(0, 1)
    y = y.swapaxes(1, 3)

    pred_y_raw = pred_y_raw[0].float().detach().cpu().numpy()
    pred_y_raw = pred_y_raw.swapaxes(0, 1)
    pred_y_raw = pred_y_raw.swapaxes(1, 3)

    pred_y = pred_y[0].float().detach().cpu().numpy()
    pred_y = pred_y.swapaxes(0, 1)
    pred_y = pred_y.swapaxes(1, 3)

    to_plot = np.array([
        x[:, :, :, :1],
        y,
        pred_y_raw,
        pred_y
    ], dtype=float)
    to_plot = to_plot.swapaxes(0, 1)

    titles = ['Input', 'GT', 'Raw Output', 'Prediction']

    print('preparing example prediction gif...')
    
    plot_and_save_gif(to_plot, gif_name, titles, verbose=True, fps=3)
    
    print('preparing metrics plots...')
    
    metrics_keys = list(history[0].keys())
    num_keys = len(metrics_keys)
    
    history_transpose = {key: [] for key in metrics_keys}
    for epoch_dict in history:
        for key, value in epoch_dict.items():
            history_transpose[key].append(value)

    for train_metric, valid_metric in zip(metrics_keys[:num_keys // 2], metrics_keys[num_keys // 2:]):
        train_vals = history_transpose[train_metric]
        valid_vals = history_transpose[valid_metric]
        
        metric = '_'.join(train_metric.split('_')[1:])
        
        print(f'plotting {metric} figure...')
        
        plot_name = f'{OUT_DIR}/{dataset.lower()}/{type(model).__name__}/{root_id}/{train_idx}_{id}/{metric}.pdf'
        plot_and_save_metric(train_vals, valid_vals, metric, plot_name)
    
    print('-'*32)
    print('done')