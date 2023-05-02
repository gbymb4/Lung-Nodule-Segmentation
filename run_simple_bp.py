import torch, random, warnings, sys

import numpy as np

from models.unet import *
from models.wnet import *
from setup import parse_config, prepare_config
from projectio import prepare_datasets, prepare_dataloaders, save_history_dict_and_model
from preprocessing import zoom_and_resize_ct_and_seg
from optim.backpropagation import SimpleBPOptimizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    warnings.simplefilter('ignore')

    config_fname = sys.argv[1]
    config_dict = parse_config(config_fname)
    config = prepare_config(config_dict)

    seed, dataset, model_type, device, optim_kwargs, loading_kwargs, dataloader_kwargs = config

    set_seed(seed)

    model = model_type(16, img_channels=2).to(device)
    
    if hasattr(model, 'compile'):
        model = model.compile()

    transforms = [zoom_and_resize_ct_and_seg]
    transform_kwargs = [{'new_size': 360}]

    datasets = prepare_datasets(
        dataset,
        transforms=transforms,
        transform_kwargs=transform_kwargs,
        load_ct_dims=[0, 1],
        **loading_kwargs
    )

    train_loader, valid_loader = prepare_dataloaders(datasets, **dataloader_kwargs)

    optim = SimpleBPOptimizer(model, train_loader, valid_loader, device=device)
    history = optim.execute(**optim_kwargs)

    print('-'*32)
    print('saving training history and model...', end='')

    train_idx = dataloader_kwargs['train_idx']

    save_history_dict_and_model(dataset, model, config_dict, train_idx, history)

    print('done')
    print('all done!')



if __name__ == '__main__':
    main()