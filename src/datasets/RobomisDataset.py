from PIL import Image, ImageFile
import numpy as np
import glob
import os
from torch.utils.data import Dataset

import torch


class Robomis(Dataset):
    def __init__(self, dir_main, split, transform = None, imsize=512):
        super(Robomis, self).__init__()
        self.transform = transform
        # self.mask_transform=mask_transform
        self.imsize = imsize
        self.img_files = glob.glob(os.path.join(dir_main,'images',split,'*.png'))
        self.mask_files = []
        self.ins_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(dir_main, 'annotations', split, os.path.basename(img_path)))
             self.ins_files.append(os.path.join(dir_main, 'annotations', split+'_instance', os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        ins_path = self.ins_files[index]
        sample = {}

     
        img = Image.open(img_path)
        img = img.convert('RGB')

        mask = Image.open(mask_path)
        mask = mask.point(lambda x: 1 if x > 0 else 0, mode='1')
            # mask = mask.convert('L')  # or mask = mask.convert('1')

        ins = Image.open(ins_path)

        if self.imsize is not None:
            img = img.resize((self.imsize, self.imsize), resample=Image.BILINEAR)
            mask = mask.resize((self.imsize, self.imsize), resample=Image.NEAREST)
            ins = ins.resize((self.imsize, self.imsize), resample=Image.NEAREST)
        # if self.transform is not None:
        #     # mat, mat_inv = self.getTransformMat(self.imsize, True)
        #     img_np = np.array(img).astype(np.uint8)
        #     mask_np = np.array(mask).astype(np.uint8)
        #     ins_np = np.array(ins).astype(np.uint8)
        #     transformed = self.transform(image=img_np, mask=mask_np, ins=ins_np)

        #     # Access the transformed image and mask
        #     # trans_img = transformed["image"]
        #     trans_img = Image.fromarray(transformed['image'].transpose(2, 0, 1)) / 255.0
        #     trans_mask = Image.fromarray(transformed["mask"])
        #     trans_ins = Image.fromarray(transformed["ins"])
        # else:
        #     trans_img = img
        #     trans_mask = mask
        #     trans_ins = ins

        sample['image'] = img
        sample['im_name'] = self.img_files[index]
        sample['instance'] = ins
        sample['label'] = mask

        if(self.transform is not None):
            return self.transform(sample)
        else:
            return sample


    
    def __len__(self):
        return len(self.img_files)