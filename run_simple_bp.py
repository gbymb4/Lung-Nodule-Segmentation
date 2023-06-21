import torch, random, warnings, sys, time, os

import numpy as np

from models.unet import *
from models.wnet import *
from setup import parse_config, prepare_config
from projectio import (
    prepare_datasets, 
    prepare_dataloaders, 
    save_history_dict_and_model, 
    plot_and_save_gif, 
    plot_and_save_metric
)
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
    print('saving training history and model...', end='')

    id = int(time.time())

    save_history_dict_and_model(dataset, model, id, config_dict, train_idx, history)

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

    for train_metric, valid_metric in zip(metrics_keys[:num_keys], metrics_keys[num_keys:-1]):
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
    dataset_type, 
    model_type, 
    device, 
    transforms, 
    model_kwargs, 
    transform_kwargs, 
    optim_kwargs, 
    loading_kwargs, 
    dataloader_kwargs,
    config_dict,
    root_id
):  
    model = model_type(**model_kwargs).to(device)
    
    if hasattr(model, 'compile'):
        model = model.compile()

    datasets = prepare_datasets(
        dataset,
        dataset_type,
        'train',
        transforms=transforms,
        transform_kwargs=transform_kwargs,
        load_ct_dims=[0, 1],
        **loading_kwargs
    )

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



def main():
    warnings.simplefilter('ignore')

    config_fname = sys.argv[1]
    config_dict = parse_config(config_fname)
    config = prepare_config(config_dict)

    seed, dataset, dataset_type, model_type, device, transforms, *rest, = config
    train, test, cross_valid, *all_kwargs = rest
    
    model_kwargs, transform_kwargs, optim_kwargs, loading_kwargs, dataloader_kwargs = all_kwargs

    set_seed(seed)

    root_id = int(time.time())
    out_root = f'{OUT_DIR}/{dataset.lower()}/{model_type.__name__}/{root_id}'
    
    if not os.path.isdir(out_root):
        os.mkdir(out_root)

    histories = []

    if train:
        if not cross_valid:
            history = run_train(
                dataset, 
                dataset_type, 
                model_type, 
                device, 
                transforms, 
                model_kwargs, 
                transform_kwargs, 
                optim_kwargs, 
                loading_kwargs, 
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
                    dataset_type, 
                    model_type, 
                    device, 
                    transforms, 
                    model_kwargs, 
                    transform_kwargs, 
                    optim_kwargs, 
                    loading_kwargs, 
                    dataloader_kwargs,
                    config_dict,
                    root_id
                )
                
                histories.append(histories)
                
    if test:
        ...
    
    print('all done!')



if __name__ == '__main__':
    main()