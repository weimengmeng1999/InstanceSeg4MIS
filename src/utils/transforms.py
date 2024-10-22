"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
# import collections
from collections.abc import Iterable
import random

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

import albumentations as A
import cv2

import torch


class CropRandomObject:

    def __init__(self, keys=[],object_key="instance", size=100):
        self.keys = keys
        self.object_key = object_key
        self.size = size

    def __call__(self, sample):

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]
        
        if unique_objects.size > 0:
            random_id = np.random.choice(unique_objects, 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)
            
            i = int(np.clip(ym-self.size[1]/2, 0, h-self.size[1]))
            j = int(np.clip(xm-self.size[0]/2, 0, w-self.size[0]))

        else:
            i = random.randint(0, h - self.size[1])
            j = random.randint(0, w - self.size[0])

        for k in self.keys:
            assert(k in sample)

            sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        return sample




class RandomCrop(T.RandomCrop):

    def __init__(self, keys=[], size=100):

        super().__init__(size)
        self.keys = keys

    def __call__(self, sample):

        params = None

        for k in self.keys:

            assert(k in sample)

            if params is None:
                params = self.get_params(sample[k], self.size)

            sample[k] = F.crop(sample[k], *params)

        return sample

class RandomRotation(T.RandomRotation):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.resample, Iterable):
            assert(len(keys) == len(self.resample))

    def __call__(self, sample):

        angle = self.get_params(self.degrees)

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            resample = self.resample
            if isinstance(resample, Iterable):
                resample = resample[idx]

            sample[k] = F.rotate(sample[k], angle, resample,
                                 self.expand, self.center)

        return sample


class Resize(T.Resize):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.interpolation, Iterable):
            assert(len(keys) == len(self.interpolation))

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, Iterable):
                interpolation = interpolation[idx]

            sample[k] = F.resize(sample[k], self.size, interpolation)

        return sample


class ToTensor(object):

    def __init__(self, keys=[], type="float"):

        if isinstance(type, Iterable):
            assert(len(keys) == len(type))

        self.keys = keys
        self.type = type

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            sample[k] = F.to_tensor(sample[k])

            t = self.type
            if isinstance(t, Iterable):
                t = t[idx]

            if t == torch.ByteTensor:
                sample[k] = sample[k]*255

            sample[k] = sample[k].type(t)

        return sample

class transvit:
    def __init__(self, keys=[],object_key="instance"):
        self.keys = keys
        self.object_key = object_key
        self.transforms = A.Compose([
                A.OneOf([
                    A.RandomSizedCrop(min_max_height=(int(
                        588 * 0.5), 588),
                                      height=588,
                                      width=588,
                                      p=0.5),
                A.PadIfNeeded(min_height=588, min_width=588, 
                              border_mode=cv2.BORDER_CONSTANT)
                ],p=1),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120,
                                       sigma=120 * 0.05,
                                       alpha_affine=120 * 0.03,
                                       p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
                ], p=0),
                        #p=0.8 if self.use_vis_aug_non_rigid else 0),
                A.CLAHE(p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
            ],
            additional_targets={'image': 'image', 'mask0': 'mask', 'mask1': 'mask'})

    def __call__(self, sample):
        transed = self.transforms(image=np.array(sample['image']).astype(np.uint8), 
                                  mask0=np.array(sample['label']).astype(np.uint8),
                                  mask1=np.array(sample['instance']).astype(np.uint8)
                                  )

        for k in self.keys:
            assert(k in sample)
            if k == 'image':
                sample[k] = transed['image']
            elif k == 'label':
                sample[k] = transed['mask0']
            elif k == 'instance':
                sample[k] = transed['mask1']

        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']
        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)

