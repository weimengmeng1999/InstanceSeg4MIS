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

    save=True,
    save_dir='./exp/orig_model/masks/',
    checkpoint_path='./exp/orig_model/best_iou_model.pth',

    dataset= { 
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
        }
    },
        
    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [3, 1],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
