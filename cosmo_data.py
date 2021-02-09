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
from torch.utils.data.dataloader import default_collate


def batchify(batch):
    data, label = default_collate(batch)
    return data.reshape(-1, 1, 64, 64), label.reshape(-1, 5)


class CosmoData(Dataset):
    def __init__(self, train='train', load_every=5):
        self.file_list = torch.load(f'/mnt/home/ecui/ceph/vae_learning/dataset_exp/{train}_list.pkl')
        img_list = [torch.from_numpy(np.load(self.file_list[ind]['data'])) for ind in range(load_every)]
        label_list = [self.file_list[ind]['cosmo pharameter'] for ind in range(load_every)]
        self.load_every = load_every
        imgs, self.labels = self.get_chunk(img_list, label_list)
        self.imgs = self.get_norm(imgs)
        del img_list

    def get_norm(self, data):
        return torch.log(data + 1e-8)

    def get_shrink(self, data, width=64, height=64):
        assert data.shape[1] >= height
        assert data.shape[2] >= width
        x = random.randint(0, data.shape[2] - width)
        y = random.randint(0, data.shape[1] - height)
        return data[:, y:y + height, x:x + width]

    def get_chunk(self, data_list, labels):
        res = []
        res_labels = []
        for j in range(len(data_list)):
            list_ = [c.chunk(8, dim=1) for c in data_list[j].chunk(8, dim=0)]
            for item in list_:
                res += [a.reshape(-1, 64, 64) for a in item]
            res_labels += [torch.from_numpy(labels[j]).repeat(8 * 8, 1)]
        return torch.cat(res, axis=0).unsqueeze(1), torch.cat(res_labels, axis=0)

    def __getitem__(self, index):
        group = index // self.load_every
        member = index % self.load_every

        if member == 0:
            true_index = [group * self.load_every + i for i in range(self.load_every)]
            img_list = [torch.from_numpy(np.load(self.file_list[ind]['data'])) for ind in true_index]
            label_list = [self.file_list[ind]['cosmo pharameter'] for ind in true_index]
            imgs, self.labels = self.get_chunk(img_list, label_list)
            self.imgs = self.get_norm(imgs)

        return self.imgs, self.labels

    def __len__(self):
        return len(self.file_list)

# Add a Dummy Dataset for test###
class DummyData(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        img = torch.rand(1, 64, 64)
        return img, torch.randn(5)

    def __len__(self):
        return 5000
