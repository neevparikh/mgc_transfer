import os
import math

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from constants import DATA_DIR

torch.multiprocessing.set_sharing_strategy('file_descriptor')
splits = [0.5, 0.3, 0.2]


def get_data(name):
    data = np.load(os.path.join(DATA_DIR, name), allow_pickle=True)
    unique_labels, labels = np.unique(data['labels'], return_inverse=True)
    spects = np.expand_dims(data['spects'], axis=1)
    print(labels.shape, labels[:10])
    print(spects.shape, spects[:10])
    return labels, spects.astype(np.float16)


class FMA(pl.LightningDataModule):
    shape = (128, 640)

    def train_dataloader(self):
        print("Number of training samples", len(self.train))
        return DataLoader(self.train, batch_size=self.batchsize, num_workers=self.num_data_workers)

    def val_dataloader(self):
        print("Number of val samples", len(self.val))
        return DataLoader(self.val, batch_size=self.batchsize, num_workers=self.num_data_workers)

    def test_dataloader(self):
        print("Number of test samples", len(self.test))
        return DataLoader(self.test, batch_size=self.batchsize, num_workers=self.num_data_workers)


class FMA_Large(FMA):
    def setup(self, args, stage=None):
        self.labels, self.spects = get_data('fma_large_combined.npz')
        self.labels = torch.from_numpy(self.labels)
        self.num_labels = 161
        self.batchsize = args.batchsize
        self.num_data_workers = args.data_workers
        self.spects = torch.from_numpy(self.spects)
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)


class FMA_Small(FMA):
    def setup(self, args, stage=None):
        self.labels, self.spects = get_data('fma_small_combined.npz')
        self.labels = torch.from_numpy(self.labels)
        self.spects = torch.from_numpy(self.spects)
        self.num_labels = 8
        self.batchsize = args.batchsize
        self.num_data_workers = args.data_workers
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)


class GTZAN(pl.LightningDataModule):
    def setup(self, args, stage=None):
        self.shape = (128, 660)
        self.num_labels = 10
        self.batchsize = args.batchsize
        self.num_data_workers = args.data_workers
        self.labels, self.spects = get_data('gtzan.npz')
        self.labels = torch.from_numpy(self.labels)
        self.spects = torch.from_numpy(self.spects)
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)

    def train_dataloader(self):
        print("Number of training samples", len(self.train))
        return DataLoader(self.train,
                          batch_size=self.batchsize,
                          pin_memory=True,
                          num_workers=self.num_data_workers)

    def val_dataloader(self):
        print("Number of val samples", len(self.val))
        return DataLoader(self.val,
                          batch_size=self.batchsize,
                          pin_memory=True,
                          num_workers=self.num_data_workers)

    def test_dataloader(self):
        print("Number of test samples", len(self.test))
        return DataLoader(self.test,
                          batch_size=self.batchsize,
                          pin_memory=True,
                          num_workers=self.num_data_workers)
