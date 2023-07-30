import yaml

from typing import Tuple, Callable
from copy import deepcopy
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
    dataset = deepcopy(config['dataset'])
    dataset_type = deepcopy(config['dataset_type'])
    model_name = deepcopy(config['model'])

    if model_name.lower() == 'unet':
        model = UNet
    elif model_name.lower() == 'r2unet':
        model = R2UNet
    elif model_name.lower() == 'r2unet360':
        model = R2UNet360
    elif model_name.lower() == 'dualr2unetenc':
        model = DualR2UNetEnc
    elif model_name.lower() == 'dualr2unet360enc':
        model = DualR2UNet360Enc
    elif model_name.lower() == 'dualr2unetdec':
        model = DualR2UNetDec
    elif model_name.lower() == 'dualr2unet360dec':
        model = DualR2UNet360Dec
    elif model_name.lower() == 'sadualr2unet':
        model = SADualR2UNet
    elif model_name.lower() == 'sadualr2unet360':
        model = SADualR2UNet360
    elif model_name.lower() == 'r2wnet':
        model = R2WNet
    else:
        raise ValueError(f'Invalid model type "{model_name}" in config file.')

    if dataset_type.lower() == 'full':
        dataset_type = LNSegDataset
    elif dataset_type.lower() == 'nodules':
        dataset_type = LNSegDatasetNodules

    device = deepcopy(config['device'])
    train = deepcopy(config['train'])
    test = deepcopy(config['test'])
    cross_valid = deepcopy(config['cross_valid'])
    root_id = deepcopy(config['root_id'])
    checkpoint_freq = deepcopy(config['checkpoint_freq'])
    
    model_kwargs = deepcopy(config['model_arguments'])
    optim_kwargs = deepcopy(config['optimizer_arguments'])
    loading_kwargs = deepcopy(config['loading_arguments'])
    dataloader_kwargs = deepcopy(config['dataloader_arguments'])

    if 'zoom_transform' in config.keys() and config['zoom_transform']:
        transforms = [zoom_and_resize_ct_and_seg]
        transform_kwargs = [{'new_size': 360}]
    else:
        transforms = None
        transform_kwargs = None

    args = (
        seed, dataset, dataset_type,
        model, device, transforms, 
        train, test, cross_valid, 
        root_id, checkpoint_freq
    )
    kwargs = (
        model_kwargs, transform_kwargs, optim_kwargs,
        loading_kwargs, dataloader_kwargs
    )

    out = (*args, *kwargs)

    return out
