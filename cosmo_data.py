import os
import glob
import sys
import numpy as np
import torch.nn as nn
import torch
import argparse
import random
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class CosmoData(Dataset):
    def __init__(self, train='train'):
        self.file_list = torch.load(f'/mnt/home/ecui/ceph/vae_learning/dataset/{train}_list.pkl')
    def get_norm(self, data):
        return torch.log(data + 1e-8)

    def get_shrink(self, data, width=64, height=64):
        assert data.shape[1] >= height
        assert data.shape[2] >= width
        x = random.randint(0, data.shape[2] - width)
        y = random.randint(0, data.shape[1] - height)
        return data[:, y:y + height, x:x + width]

    def __getitem__(self, index):
        img = self.file_list[index]['data']
        img = torch.from_numpy(np.load(img)).view(-1, 512, 512)
        img = self.get_shrink(self.get_norm(img))
        return img, self.file_list[index]['cosmo pharameter']

    def __len__(self):
        return len(self.file_list)

### Add a Dummy Dataset for test###
class DummyData(Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        img = torch.rand(1, 64, 64)
        return img, torch.randn((1, 1))
    
    def __len__(self):
        return 153 * 2000
