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
    plot_and_save_slide,
    rank_histories,
    last_checkpoint,
    dump_preds_gif_and_metrics_plots
)

from optim import compute_all_metrics
from optim.backpropagation import SimpleBPOptimizer

from putils import evaluate_model

from pconfig import (
    OUT_DIR,
    LUNA16_NUM_SUBSETS,
    NSCLC_NUM_SUBSETS
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    

def run_train(
    seed,
    dataset, 
    model_type, 
    device, 
    model_kwargs, 
    optim_kwargs, 
    datasets,
    dataloader_kwargs,
    config_dict,
    root_id,
    checkpoint_freq
):  
    model = model_type(**model_kwargs).to(device)
    
    if hasattr(model, 'compile'):
        model = model.compile()

    train_loader, valid_loader = prepare_dataloaders(datasets, **dataloader_kwargs)

    id = int(time.time())
    train_idx = dataloader_kwargs['train_idx']
    
    last_epoch, last_id, last_history = last_checkpoint(
        dataset, 
        model, 
        root_id, 
        train_idx
    )
    
    if last_id is not None:
        id = last_id

    def checkpoint(hist, epoch):
        if epoch % checkpoint_freq == 0:
            dump_preds_gif_and_metrics_plots(
                dataset, 
                model, 
                hist, 
                valid_loader, 
                device, 
                train_idx, 
                id,
                root_id,
                config_dict
            )
        
    optim = SimpleBPOptimizer(seed, model, train_loader, valid_loader, device=device)
    history = optim.execute(
        **optim_kwargs, 
        checkpoint_callback=checkpoint, 
        start_epoch=last_epoch,
        init_history=last_history
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
    root_id,
):

    train_idx, id = rank_histories(model_type, root_id)
    
    test_dir = '{}/{}/{}/{}/{}_{}/testing'.format(
        OUT_DIR,
        dataset.lower(),
        model_type.__name__,
        root_id,
        train_idx,
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
    
    model = load_model(
        model_type, 
        model_kwargs, 
        root_id, 
        train_idx,
        id
    )
    model = model.to(device)
    
    wavg_metrics = evaluate_model(
        model,
        testloader, 
        device, 
        cum_batch_size, 
        gifs_dir, 
        plots_dir, 
        plot_and_save_gif, 
        plot_and_save_slide
    )
    
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

    seed, dataset, dataset_type, model_type, device, transforms, *rest, = config
    train, test, cross_valid, root_id, checkpoint_freq, *all_kwargs = rest
    
    model_kwargs, transform_kwargs, optim_kwargs, loading_kwargs, dataloader_kwargs = all_kwargs

    set_seed(seed)

    if root_id is None:
        root_id = int(time.time())
    
    out_root = f'{OUT_DIR}/{dataset.lower()}/{model_type.__name__}/{root_id}'
    
    if not os.path.isdir(out_root):
        os.mkdir(out_root)

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
            run_train(
                seed,
                dataset, 
                model_type, 
                device, 
                model_kwargs, 
                optim_kwargs, 
                datasets,
                dataloader_kwargs,
                config_dict,
                root_id,
                checkpoint_freq
            )
            
        else:
            if dataset.lower() == 'luna16':
                num_subsets = LUNA16_NUM_SUBSETS
            elif dataset.lower() == 'nsclc':
                num_subsets = NSCLC_NUM_SUBSETS
            else:
                raise ValueError(f"invalid value for arg 'dataset': {dataset}")
                
            for train_idx in range(num_subsets):
                dataloader_kwargs['train_idx'] = train_idx
                
                run_train(
                    seed,
                    dataset, 
                    model_type, 
                    device, 
                    model_kwargs, 
                    optim_kwargs, 
                    datasets,
                    dataloader_kwargs,
                    config_dict,
                    root_id,
                    checkpoint_freq
                )
                
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
            root_id
        )
    
    print('all done!')



if __name__ == '__main__':
    main()