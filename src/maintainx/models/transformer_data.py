"""Dataset and DataModule for the windowed RUL transformer."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from maintainx.models.utils import _load_features, split_by_unit
from maintainx.features.windowing import make_train_windows, make_last_window


class WindowDataset(Dataset):
    def __init__(self, X, y, rul_cap=125):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.clip(y, 0, rul_cap)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RULWindowDataModule(L.LightningDataModule):
    def __init__(self, window=30, stride=5, batch_size=64, val_fraction=0.2, seed=42):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        train_features, test_features = _load_features()
        train_split, val_split = split_by_unit(
            train_features, self.hparams.val_fraction, self.hparams.seed
        )

        X_train, y_train = make_train_windows(
            train_split, window=self.hparams.window, stride=self.hparams.stride
        )
        X_val, y_val, _ = make_last_window(val_split, window=self.hparams.window)
        X_test, y_test, self.test_unit_ids = make_last_window(
            test_features, window=self.hparams.window
        )

        # Standardize features using TRAIN statistics only (avoid leakage)
        self.feat_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        self.feat_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-6

        self.train_ds = WindowDataset(self._normalize(X_train), y_train)
        self.val_ds = WindowDataset(self._normalize(X_val), y_val)
        self.test_ds = WindowDataset(self._normalize(X_test), y_test)
        self.n_features = X_train.shape[-1]

    def _normalize(self, X):
        return (X - self.feat_mean) / self.feat_std

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=0)