#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torchvision import transforms
from torch.utils.data import Dataset, Sampler, DataLoader
from glob import glob
from tqdm import tqdm
import os.path as osp
import pandas as pd
from PIL import Image
import random
import cv2
import operator
from functools import reduce
import torch
import numpy as np
try:
    from src.config import conf
except:
    from config import conf


class PreLoadData(Dataset):
    def __init__(self, font_size=60, transform=None, subset="train"):
        self.fonts = 28
        self.str_fount = 0
        self.end_fount = conf.num_fonts
        if subset == "train":
            self.str_fount = 0
            self.end_fount = conf.num_fonts
            self.fonts = self.end_fount - self.str_fount
        elif subset == "val":
            self.str_fount = conf.num_fonts
            self.end_fount = 28
            self.fonts = self.end_fount - self.str_fount
        self.protype_paths = glob(osp.join(conf.folder,'data','normal', "*.png"))
        self.protype_paths.sort()
        
        self.protype_imgs = [None for i in range(len(self.protype_paths))]
        for i in range(len(self.protype_paths)):
            img = Image.open(self.protype_paths[i]).convert('RGB') 
            img = np.array(img) 
            img = img[:, :, ::-1].copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(img)
            self.protype_imgs[i] = img

        self.img_path_dict = {}
        fount_index = 0
        for fount_path in range(self.str_fount,self.end_fount):
            fount_img = glob(osp.join(conf.folder,'data',str(fount_path), "*.png"))
            fount_img.sort()
            self.img_path_dict[fount_index] = fount_img
            fount_index += 1
            
        self.img_dict = {}

        for k, v in self.img_path_dict.items():
            tmp = []
            for i in range(conf.num_chars):
                img = Image.open(v[i])
                img = np.array(img) 
                img = img[:, :, ::-1].copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = Image.fromarray(img)
                tmp.append(img)
            self.img_dict[k] = tmp

        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((64, 64)), transforms.ToTensor()]
            )
        else:
            self.transform = transform
        if conf.custom_batch:
            self.style_label = torch.tensor(
                    reduce(
                        operator.add,
                        [
                            [j for i in range(conf.num_chars)]
                            for j in range(self.fonts)
                        ],
                    )
                )

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        font_index = index // conf.num_chars
        char_index = index % conf.num_chars
        protype_img = self.protype_imgs[char_index]

        real_img = self.img_dict[font_index][char_index]

        style_char_index = random.randint(0, conf.num_chars - 1)
        style_img = self.img_dict[font_index][style_char_index]
        protype_img = self.transform(protype_img)
        real_img = self.transform(real_img)
        style_img = self.transform(style_img)

        return (
            protype_img,
            char_index,
            style_img,
            font_index,
            style_char_index,
            real_img,
        )

    def __len__(self):
        return self.fonts * conf.num_chars - 1


class CustomSampler(Sampler):
    def __init__(self, data, shuffle=True):
        self.data = data
        self.shuffle = shuffle

    def __iter__(self):
        indices = []
        font_indices = [i for i in range(self.data.fonts)]
        if self.shuffle:
            random.shuffle(font_indices)
        # for n in range(conf.num_fonts):
        for n in font_indices:
            index = torch.where(self.data.style_label == n)[0]
            #idx = torch.randperm(index.nelement())
            #index = index.view(-1)[idx].view(index.size())
            indices.append(index)
        indices = torch.cat(indices, dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)


class CustomBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(sampler_list) - 1
                and self.sampler.data.style_label[idx]
                != self.sampler.data.style_label[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":

    train_data = PreLoadData()
    
    train_sampler = CustomSampler(train_data, shuffle=True)
    train_batch_sampler = CustomBatchSampler(
        train_sampler, conf.batch_size, drop_last=False
    )
    
    train_dl = DataLoader(train_data, batch_sampler=train_batch_sampler)
    from tqdm import tqdm
    topil = transforms.ToPILImage()
    for i in tqdm(train_dl, total=2):
        print(i[4])

        

