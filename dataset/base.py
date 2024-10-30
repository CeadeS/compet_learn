
from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform = None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1 : im = im.convert('RGB') 
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]

from tqdm import tqdm
class BaseTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, root, classes, transform = None):
        self.classes = classes
        self.root = root
        self.data = dataset(root=root, download=False)
        self.transform = transform
        self.ys, self.I, self.indexes = [], [], {}
        index = 0
        for idx, (_, y) in tqdm(enumerate(self.data)):
            if y in classes: # choose only specified classes
                self.ys.append(y)
                self.I += [index]
                self.indexes[index] = idx
                index += 1
        print()
        print()
        print("Number of samples:", len(self.I))

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        mapped_index = self.indexes[index]
        im, y = self.data[mapped_index]
        if isinstance(im, torch.Tensor):
            im = torchvision.transforms.functional.to_pil_image(im)
        # convert gray to rgb
        if len(list(im.split())) == 1 : im = im.convert('RGB') 
        if self.transform is not None:
            im = self.transform(im)            
        assert y == self.ys[index]
        
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
