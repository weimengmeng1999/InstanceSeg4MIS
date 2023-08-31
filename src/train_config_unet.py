"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

ROBOMIS_DIR=os.environ.get('ROBOMIS_DIR')

args = dict(

    cuda=True,
    display=True,
    display_it=5,

    save=True,
    save_dir='./exp/orunet_1000e',
    resume_path=None, 

    train_dataset = {
        'name': 'robomis',
        'kwargs': {
            'dir_main': ROBOMIS_DIR,
            'split': 'training',
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'size': (512, 512),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 16,
        'workers': 8
    }, 

    val_dataset = {
        'name': 'robomis',
        'kwargs': {
            'dir_main': ROBOMIS_DIR,
            'split': 'validation',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 16,
        'workers': 8
    }, 

    model = {
        'name': 'branched_orunet', 
        'kwargs': {}
    }, 

    lr=5e-4,
    n_epochs=1000,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': 1,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
)


def get_args():
    return copy.deepcopy(args)
