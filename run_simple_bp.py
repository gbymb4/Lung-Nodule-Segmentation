import torch, random, warnings

import numpy as np

from models.unet import *
from models.wnet import *
from projectio import prepare_datasets, prepare_dataloaders
from preprocessing import zoom_and_resize_ct_and_seg
from optim.backpropagation import SimpleBPOptimizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    warnings.simplefilter('ignore')
    set_seed(0)

    model = R2UNet(16)

    transforms = [zoom_and_resize_ct_and_seg]
    transform_kwargs = [{'new_size': (360, 360)}]

    datasets = prepare_datasets(
        'LUNA16',
        transforms=transforms,
        transform_kwargs=transform_kwargs,
        load_ct_dims=[0, 1]
    )

    train_loader, valid_loader = prepare_dataloaders(datasets, train_idx=0, batch_size=8)

    optim = SimpleBPOptimizer(model, train_loader, valid_loader, device='cuda')
    history = optim.execute()

    print('all done!')



if __name__ == '__main__':
    main()