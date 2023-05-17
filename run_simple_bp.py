import torch, random, warnings, sys, time

import numpy as np

from models.unet import *
from models.wnet import *
from setup import parse_config, prepare_config
from projectio import prepare_datasets, prepare_dataloaders, save_history_dict_and_model, plot_and_save_gif
from optim.backpropagation import SimpleBPOptimizer
from pconfig import OUT_DIR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    warnings.simplefilter('ignore')

    config_fname = sys.argv[1]
    config_dict = parse_config(config_fname)
    config = prepare_config(config_dict)

    seed, dataset, dataset_type, model_type, device, transforms, *all_kwargs = config
    transform_kwargs, optim_kwargs, loading_kwargs, dataloader_kwargs = all_kwargs

    set_seed(seed)

    model = model_type(16, img_channels=2).to(device)
    
    if hasattr(model, 'compile'):
        model = model.compile()

    datasets = prepare_datasets(
        dataset,
        dataset_type,
        transforms=transforms,
        transform_kwargs=transform_kwargs,
        load_ct_dims=[0, 1],
        **loading_kwargs
    )

    train_loader, valid_loader = prepare_dataloaders(datasets, **dataloader_kwargs)

    optim = SimpleBPOptimizer(model, train_loader, valid_loader, device=device)
    history = optim.execute(**optim_kwargs)

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

    train_idx = dataloader_kwargs['train_idx']

    id = int(time.time())

    save_history_dict_and_model(dataset, model, id, config_dict, train_idx, history)

    gif_name = f'{OUT_DIR}/{dataset.lower()}/{type(model).__name__}/{train_idx}_{id}/example_preds.gif'

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
    print('-'*32)

    plot_and_save_gif(to_plot, gif_name, titles, verbose=True, fps=3)

    print('done')
    print('all done!')



if __name__ == '__main__':
    main()