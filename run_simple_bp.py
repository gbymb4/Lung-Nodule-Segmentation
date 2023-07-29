import warnings, sys, os, json
import math, torch, random, time

import numpy as np

from models.unet import *
from models.wnet import *

from setup import parse_config, prepare_config

from projectio import (
    prepare_datasets, 
    prepare_dataloaders, 
    prepare_testloader,
    load_model,
    save_history_dict_and_model, 
    plot_and_save_gif, 
    plot_and_save_metric,
    plot_and_save_slide
)

from optim import compute_all_metrics
from optim.backpropagation import SimpleBPOptimizer

from pconfig import (
    OUT_DIR,
    LUNA16_NUM_SUBSETS,
    NSCLC_NUM_SUBSETS
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    

def dump_preds_gif_and_metrics_plots(
    dataset, 
    model, 
    history,
    valid_loader, 
    device, 
    train_idx, 
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

    id = int(time.time())

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
    


def run_train(
    dataset, 
    model_type, 
    device, 
    model_kwargs, 
    optim_kwargs, 
    datasets,
    dataloader_kwargs,
    config_dict,
    root_id
):  
    model = model_type(**model_kwargs).to(device)
    
    if hasattr(model, 'compile'):
        model = model.compile()

    train_loader, valid_loader = prepare_dataloaders(datasets, **dataloader_kwargs)

    optim = SimpleBPOptimizer(model, train_loader, valid_loader, device=device)
    history = optim.execute(**optim_kwargs)

    train_idx = dataloader_kwargs['train_idx']

    dump_preds_gif_and_metrics_plots(
        dataset, 
        model, 
        history, 
        valid_loader, 
        device, 
        train_idx, 
        root_id,
        config_dict
    )
    
    return history



def run_test(
    dataset,
    dataset_type,
    model_type,
    model_kwargs,
    device,
    loading_kwargs,
    transforms,
    transform_kwargs,
    dataloader_kwargs,
    cum_batch_size,
    histories,
    root_id,
):
    testset = prepare_datasets(
        dataset,
        dataset_type,
        'test',
        transforms=transforms,
        transform_kwargs=transform_kwargs,
        load_ct_dims=[0, 1],
        **loading_kwargs
    )[0]
    
    testloader = prepare_testloader(testset, **dataloader_kwargs)
    
    last_training_epochs_dice = [history[-1]['valid_hard_dice'] for history in histories]
    best_train_idx_model = np.argmax(last_training_epochs_dice)
    
    model = load_model(
        model_type, 
        model_kwargs, 
        root_id, 
        best_train_idx_model,
        id
    )
    model = model.to(device)
    
    metrics_dict = {}
    num_slides = 0
    
    render_freq = 10
    
    test_dir = '{}/{}/{}/{}/{}_{}_test'.format(
        OUT_DIR,
        dataset.lower(),
        type(model).__name__,
        root_id,
        best_train_idx_model,
        id
    )
    
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
        
    gifs_dir = f'{test_dir}/gifs'
    
    if not os.path.isdir(gifs_dir):
        os.mkdir(gifs_dir)
        
    plots_dir = f'{test_dir}/plots'
    
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)

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
                
            for ct_chunk, seg_chunk in zip(ct_chunks, seg_chunks):
                ct_chunk = ct_chunk.unsqueeze(dim=0).float()
                seg_chunk = seg_chunk.unsqueeze(dim=0).float()
                
                inputs.append(ct_chunk.detach().cpu().numpy()[0])
                gts.append(seg_chunk.detach().cpu().numpy()[0])

                pred_chunk = model(ct_chunk)
                predictions.append(pred_chunk.detach().cpu().numpy())
                
                batch_size = ct_chunk.shape[1]
                num_slides += batch_size
                
                metric_scores = compute_all_metrics(pred_chunk, seg_chunk)
                
                for name, score in metric_scores.items():
                    if name not in metrics_dict.keys():
                        metrics_dict[name] = score * batch_size
                    else:
                        metrics_dict[name] += score * batch_size
                    
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
                plot_and_save_gif(to_plot, gif_name, titles, verbose=True, fps=3)
                
                print('plotting middle slides...')
                
                plot_name = f'{plots_dir}/' + 'example_{}_plot_{}.pdf'
                plot_and_save_slide(inputs[len(inputs) // 2], plot_name.format('input', scan_count))
                plot_and_save_slide(inputs[len(gts) // 2], plot_name.format('gt', scan_count))
                plot_and_save_slide(inputs[len(predictions) // 2], plot_name.format('raw_out', scan_count))
                plot_and_save_slide(inputs[len(hard_predictions) // 2], plot_name.format('pred', scan_count))
                
            scan_count += 1
    
    wavg_metrics = {
        f'test_{name}': w_score / num_slides for name, w_score in metrics_dict.items()
    }
    
    metrics_fname = f'{test_dir}/test_performance.json'
    
    with open(metrics_fname, 'w') as file:
        json.dump(wavg_metrics, file)
    
    print('done!')
    print('-'*32)
    
    


def main():
    warnings.simplefilter('ignore')

    config_fname = sys.argv[1]
    config_dict = parse_config(config_fname)
    config = prepare_config(config_dict)

    seed, dataset, dataset_type, model_type, device, transforms, root_id, *rest, = config
    train, test, cross_valid, *all_kwargs = rest
    
    model_kwargs, transform_kwargs, optim_kwargs, loading_kwargs, dataloader_kwargs = all_kwargs

    set_seed(seed)

    if root_id is None:
        root_id = int(time.time())
    
    out_root = f'{OUT_DIR}/{dataset.lower()}/{model_type.__name__}/{root_id}'
    
    if not os.path.isdir(out_root):
        os.mkdir(out_root)

    histories = []

    if train:
        datasets = prepare_datasets(
            dataset,
            dataset_type,
            'train',
            transforms=transforms,
            transform_kwargs=transform_kwargs,
            load_ct_dims=[0, 1],
            **loading_kwargs
        )
        
        if not cross_valid:
            history = run_train(
                dataset, 
                model_type, 
                device, 
                model_kwargs, 
                optim_kwargs, 
                datasets,
                dataloader_kwargs,
                config_dict,
                root_id
            )
            
            histories.append(history)
            
        else:
            if dataset.lower() == 'luna16':
                num_subsets = LUNA16_NUM_SUBSETS
            elif dataset.lower() == 'nsclc':
                num_subsets = NSCLC_NUM_SUBSETS
            else:
                raise ValueError(f"invalid value for arg 'dataset': {dataset}")
                
            for train_idx in range(num_subsets):
                dataloader_kwargs['train_idx'] = train_idx
                
                history = run_train(
                    dataset, 
                    model_type, 
                    device, 
                    model_kwargs, 
                    optim_kwargs, 
                    datasets,
                    dataloader_kwargs,
                    config_dict,
                    root_id
                )
                
                histories.append(histories)
                
    if test:
        cum_batch_size = dataloader_kwargs['cum_batch_size']
        
        run_test(
            dataset, 
            dataset_type, 
            model_type, 
            model_kwargs, 
            loading_kwargs, 
            transforms,
            transform_kwargs,
            dataloader_kwargs, 
            cum_batch_size,
            histories,
            root_id
        )
    
    print('all done!')



if __name__ == '__main__':
    main()