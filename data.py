import os 

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import pandas as pd

DATA_DIR = './data'

class FMA_Small(pl.LightningDataModule):
    def prepare_data(self):
        self.df = pd.read_csv(os.path.join(DATA_DIR, 'fma_small_combined.csv'))

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

class FMA_Large(pl.LightningDataModule):
    def prepare_data(self):
        self.df = pd.read_csv(os.path.join(DATA_DIR, 'fma_large_combined.csv'))

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

class GTZAN(pl.LightningDataModule):
    def prepare_data(self):
        df = pd.read_csv(os.path.join(DATA_DIR, 'gtzan.csv'))

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
