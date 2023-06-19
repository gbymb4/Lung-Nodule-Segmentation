import yaml

from typing import Tuple, Callable
from models.unet import *
from models.wnet import *
from projectio import LNSegDataset, LNSegDatasetNodules
from preprocessing import zoom_and_resize_ct_and_seg

def parse_config(fname: str) -> dict:
    with open(fname, 'r') as file:
        config = yaml.safe_load(file)

    return config



def prepare_config(
    config: dict
) -> Tuple[int, str, Callable, str, dict, dict, dict]:
    seed = config['seed']
    dataset = config['dataset']
    dataset_type = config['dataset_type']
    model_name = config['model']

    if model_name.lower() == 'r2unet':
        model = R2UNet
    if model_name.lower() == 'unet':
        model = UNet
    elif model_name.lower() == 'r2wnet':
        model = R2WNet
    else:
        raise ValueError(f'Invalid model type "{model_name}" in config file.')

    if dataset_type.lower() == 'full':
        dataset_type = LNSegDataset
    elif dataset_type.lower() == 'nodules':
        dataset_type = LNSegDatasetNodules

    device = config['device']
    train = config['train']
    test = config['test']
    
    optim_kwargs = config['optimizer_arguments']
    loading_kwargs = config['loading_arguments']
    dataloader_kwargs = config['dataloader_arguments']

    if 'zoom_transform' in config.keys() and config['zoom_transform']:
        transforms = [zoom_and_resize_ct_and_seg]
        transform_kwargs = [{'new_size': 360}]
    else:
        transforms = None
        transform_kwargs = None

    args = (seed, dataset, dataset_type, model, device, transforms, train, test)
    kwargs = (transform_kwargs, optim_kwargs, loading_kwargs, dataloader_kwargs)

    out = (*args, *kwargs)

    return out
