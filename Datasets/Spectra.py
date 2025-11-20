from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from settings import SEED

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = Path("Datasets/Data/data.npz")
SAMPLE_PATH = Path("Datasets/Samples/Spectra.png")

FREQ_RANGE = (2500.0, 3200.0)


def load_raw(shuffle=False):
    data = np.load(DATA_PATH)
    X, y = data["X"], data["y"]
    assert len(X) == len(y), "Mismatched number of X and y"
    if shuffle:
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
    return X, y


def load_dataset(shuffle=False, test_ratio=0.2):
    X, y = load_raw(shuffle=shuffle)
    X = torch.from_numpy(X)  # (N, L)
    y = torch.from_numpy(y)  # (N, 3)
    full_ds = TensorDataset(X, y)
    train_idx = int(len(full_ds) * (1 - test_ratio))
    test_idx = len(full_ds) - train_idx
    train_ds = torch.utils.data.Subset(full_ds, range(0, train_idx))
    test_ds = torch.utils.data.Subset(full_ds, range(train_idx, len(full_ds)))
    return train_ds, test_ds


def load_loader(shuffle=False, test_ratio=0.2, batch_size=64):
    train_ds, test_ds = load_dataset(shuffle=shuffle, test_ratio=test_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader


def plot():
    sample_loader, _ = load_loader(shuffle=True, test_ratio=0.2, batch_size=16)
    X, y = next(iter(sample_loader))
    X, y = X.numpy(), y.numpy()
    fig, axes = plt.subplots(4, 1, figsize=(9, 16))
    for i in range(4):
        axes[i].set_title(
            rf"$\mathbf{{B}}={y[i,0]:.2f}, \theta={y[i,1]:.2f}, \phi={y[i,2]:.2f}$",
            fontsize=16,
        )
        axes[i].plot(X[i])
        axes[i].set_xlabel("Frequency (MHz)", fontsize=14)
        axes[i].set_ylabel("Normalized Signal", fontsize=14)
    fig.suptitle(f"X: {X.shape}, y: {y.shape}", y=0.99, fontsize=16)

    plt.tight_layout()
    plt.savefig(SAMPLE_PATH)
    plt.close(fig)


if __name__ == "__main__":
    plot()
