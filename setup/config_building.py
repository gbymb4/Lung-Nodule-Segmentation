import yaml

from typing import Tuple, Callable
from models.unet import *
from models.wnet import *

def parse_config(fname: str) -> dict:
    with open(fname, 'r') as file:
        config = yaml.safe_load(file)

    return config



def prepare_config(
    config: dict
) -> Tuple[int, str, Callable, str, dict, dict, dict]:
    seed = config['seed']
    dataset = config['dataset']
    model_name = config['model']

    if model_name.lower() == 'r2unet':
        model = R2UNet
    elif model_name.lower() == 'r2wnet':
        model = R2WNet
    else:
        raise ValueError(f'Invalid model type "{model_name}" in config file.')

    device = config['device']
    optim_kwargs = config['optimizer_arguments']
    loading_kwargs = config['loading_arguments']
    dataloader_kwargs = config['dataloader_arguments']

    return seed, dataset, model, optim_kwargs, loading_kwargs, dataloader_kwargs
