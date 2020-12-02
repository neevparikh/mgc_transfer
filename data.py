import os
import math

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from constants import DATA_DIR, BATCHSIZE

splits = [0.5, 0.3, 0.2]

def get_data(name):
    data = np.load(os.path.join(DATA_DIR, name), allow_pickle=True)
    unique_labels, labels = np.unique(data['labels'], return_inverse=True)
    spects = np.expand_dims(data['spects'], axis=1) 
    return labels, spects.astype(np.float16)

class FMA(pl.LightningDataModule):
    shape = (128, 640)
    num_labels = 8

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCHSIZE)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=BATCHSIZE)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCHSIZE)

class FMA_Large(FMA):
    def setup(self, stage=None):
        self.labels, self.spects = get_data('fma_large_combined.npz')
        self.labels = torch.from_numpy(self.labels)
        self.spects = torch.from_numpy(self.spects)
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)

class FMA_Small(FMA):
    def setup(self, stage=None):
        self.labels, self.spects = get_data('fma_small_combined.npz')
        self.labels = torch.from_numpy(self.labels)
        self.spects = torch.from_numpy(self.spects)
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)

class GTZAN(pl.LightningDataModule):
    def setup(self, stage=None):
        self.shape = (128, 660)
        self.num_labels = 10

        self.labels, self.spects = get_data('gtzan.npz')
        self.labels = torch.from_numpy(self.labels)
        self.spects = torch.from_numpy(self.spects)
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCHSIZE, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=BATCHSIZE, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCHSIZE, pin_memory=True)
