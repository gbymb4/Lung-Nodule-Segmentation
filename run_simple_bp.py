import torch, random, warnings

import numpy as np

from models.unet import *
from models.wnet import *
from projectio import prepare_datasets, prepare_dataloaders, save_history_dict_and_model
from preprocessing import zoom_and_resize_ct_and_seg
from optim.backpropagation import SimpleBPOptimizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    warnings.simplefilter('ignore')
    set_seed(0)

    device = 'cuda'
    dataset = 'LUNA16'
    train_idx = 0

    model = R2UNet(16, img_channels=2).to(device)
    
    if hasattr(model, 'compile'):
        model = model.compile()

    transforms = [zoom_and_resize_ct_and_seg]
    transform_kwargs = [{'new_size': 360}]

    datasets = prepare_datasets(
        'LUNA16',
        transforms=transforms,
        transform_kwargs=transform_kwargs,
        load_ct_dims=[0, 1]
    )

    train_loader, valid_loader = prepare_dataloaders(datasets, train_idx=train_idx, batch_size=8)

    optim = SimpleBPOptimizer(model, train_loader, valid_loader, device=device)
    history = optim.execute(cum_batch_size=32)

    print('-'*32)
    print('saving training history and model...', end='')

    save_history_dict_and_model(dataset, model, train_idx, history)

    print('done')
    print('all done!')



if __name__ == '__main__':
    main()