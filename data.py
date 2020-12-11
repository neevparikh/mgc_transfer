import os
import math

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from constants import DATA_DIR
from utils import genre_to_label

torch.multiprocessing.set_sharing_strategy('file_descriptor')
splits = [0.5, 0.3, 0.2]
genre_to_label = np.vectorize(genre_to_label)

def get_data(name, dtype, to_keep=None):
    data = np.load(os.path.join(DATA_DIR, name), allow_pickle=True)
    spects, labels = data['spects'], data['labels']
    if to_keep:
        idxs = np.isin(labels, to_keep)
        labels = labels[idxs]
        spects = spects[idxs]
    labels = genre_to_label(labels)
    spects = np.expand_dims(spects, axis=1)
    print(labels.shape, labels[:2])
    print(spects.shape, spects[:2])
    return labels, spects.astype(dtype)


class FMA(pl.LightningDataModule):
    shape = (128, 640)

    def __init__(self, args):
        self.args = args
        self.batchsize = args.batchsize
        self.num_data_workers = args.data_workers
        self.dtype = np.float16 if args.num_gpus > 0 else np.float32

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
    def setup(self, stage=None):
        if stage == 'fit':
            self.labels, self.spects = get_data('fma_large_combined.npz', self.dtype,
                    self.args.include)
        if stage == 'test':
            self.labels, self.spects = get_data('fma_large_combined.npz', self.dtype)
        self.labels = torch.from_numpy(self.labels)
        self.num_labels = 161
        self.spects = torch.from_numpy(self.spects)
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)


class FMA_Small(FMA):
    def setup(self, stage=None):
        if stage == 'fit':
            self.labels, self.spects = get_data('fma_small_combined.npz', self.dtype,
                    self.args.include)
        if stage == 'test':
            self.labels, self.spects = get_data('fma_small_combined.npz', self.dtype)
        self.labels = torch.from_numpy(self.labels)
        self.spects = torch.from_numpy(self.spects)
        self.num_labels = 8
        self.dataset = TensorDataset(self.spects, self.labels)
        split = [math.floor(len(self.dataset) * x) for x in splits]
        split[-1] += len(self.dataset) - sum(split)
        self.train, self.val, self.test = random_split(self.dataset, split, generator=None)


class GTZAN(pl.LightningDataModule):
    def __init__(self, args):
        self.args = args
        self.batchsize = args.batchsize
        self.num_data_workers = args.data_workers
        self.dtype = np.float16 if args.num_gpus > 0 else np.float32

    def setup(self, stage=None):
        self.shape = (128, 660)
        self.num_labels = 10
        if stage == 'fit':
            self.labels, self.spects = get_data('gtzan.npz', self.dtype, self.args.include)
        if stage == 'test':
            self.labels, self.spects = get_data('gtzan.npz', self.dtype)
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
