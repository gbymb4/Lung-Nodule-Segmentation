# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:06:02 2023

@author: Gavin
"""

import torch, math

import numpy as np

from optim import compute_all_metrics
from putils import compute_centroids, centroid_slides

def evaluate_model(
    model,
    testloader, 
    device, 
    cum_batch_size,
    gifs_dir,
    plots_dir,
    gif_plotting_callback,
    fig_plotting_callback
):
    metrics_dict = {}
    num_slides = 0
    num_centroid_slides = 0
    
    render_freq = 10
    
    print('-'*32)
    print('beginning test loop...')

    scan_count = 0
    for batch in testloader:
        xs, ys = batch
            
        for ct, seg in zip(xs, ys):
            num_chunks = math.ceil(ct.size(1) / cum_batch_size)
                
            ct = ct.to(device)
            seg = seg.to(device)
            
            ct_chunks = torch.chunk(ct, num_chunks, dim=1)
            seg_chunks = torch.chunk(seg, num_chunks, dim=1)
            
            if len(ct_chunks) > 1 and ct_chunks[-1].size(1) == 1:
                merged_ct_chunk = torch.cat((ct_chunks[-2], ct_chunks[-1]), dim=1)
                ct_chunks = ct_chunks[:-2] + (merged_ct_chunk,)
                
                merged_seg_chunk = torch.cat((seg_chunks[-2], seg_chunks[-1]), dim=1)
                seg_chunks = seg_chunks[:-2] + (merged_seg_chunk,)
                
            inputs = []
            gts = []
            predictions = []
            
            pred_chunks = []
                
            for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                ct_chunk = ct_chunk.unsqueeze(dim=0).float()
                seg_chunk = seg_chunk.unsqueeze(dim=0).float()
                
                inputs.append(ct_chunk.detach().cpu().numpy()[0])
                gts.append(seg_chunk.detach().cpu().numpy()[0])

                pred_chunk = model(ct_chunk)
                predictions.append(pred_chunk.detach().cpu().numpy())
                pred_chunks.append(pred_chunk)
                
                batch_size = ct_chunk.shape[1]
                num_slides += batch_size
                
                metric_scores = compute_all_metrics(pred_chunk, seg_chunk)
                
                for name, score in metric_scores.items():
                    if name not in metrics_dict.keys():
                        metrics_dict[name] = score * batch_size
                    else:
                        metrics_dict[name] += score * batch_size
                    
            pred = torch.cat(pred_chunks, dim=0)
            
            centroids = compute_centroids(seg)
            centroids_size = len(centroids)
            num_centroid_slides += centroids_size
            
            seg_centroid_slides = centroid_slides(seg, centroids)
            pred_centroid_slices = centroid_slides(pred, centroids)
            
            centroid_metric_scores = compute_all_metrics(
                pred_centroid_slices, 
                seg_centroid_slides
            )
            
            for name, score in centroid_metric_scores.items():
                name = f'{name}_centroid'
                
                if name not in metrics_dict.keys():
                    metrics_dict[name] = score * centroids_size
                else:
                    metrics_dict[name] += score * centroids_size 
                    
            if scan_count % render_freq == 0:
                inputs = np.concatenate(inputs, axis=1)
                inputs = inputs.swapaxes(0, 1)
                inputs = inputs.swapaxes(1, 3)
                
                gts = np.concatenate(gts, axis=1)
                gts = gts.swapaxes(0, 1)
                gts = gts.swapaxes(1, 3)
                
                predictions = np.concatenate(predictions, axis=1)
                predictions = predictions.swapaxes(0, 1)
                predictions = predictions.swapaxes(1, 3)
                
                hard_predictions = predictions > 0.5
                
                to_plot = np.array([
                    inputs[:, :, :, :1],
                    gts,
                    predictions,
                    hard_predictions
                ], dtype=float)
                to_plot = to_plot.swapaxes(0, 1)

                titles = ['Input', 'GT', 'Raw Output', 'Prediction']

                print('preparing example prediction gif...')
                
                gif_name = f'{gifs_dir}/example_preds_gif_{scan_count}.gif'
                gif_plotting_callback(to_plot, gif_name, titles, verbose=True, fps=3)
                
                print('plotting middle slides...')
                
                plot_name = f'{plots_dir}/' + 'example_{}_plot_{}.pdf'
                fig_plotting_callback(inputs[len(inputs) // 2], plot_name.format('input', scan_count))
                fig_plotting_callback(inputs[len(gts) // 2], plot_name.format('gt', scan_count))
                fig_plotting_callback(inputs[len(predictions) // 2], plot_name.format('raw_out', scan_count))
                fig_plotting_callback(inputs[len(hard_predictions) // 2], plot_name.format('pred', scan_count))
                
            scan_count += 1
    
    wavg_metrics = {}
    for name, w_score in metrics_dict.items():
        if 'centroid' not in name:
            wavg_metrics[f'test_{name}'] = w_score / num_slides
        
        else:
            wavg_metrics[f'test_{name}'] = w_score / num_centroid_slides
    
    return wavg_metrics