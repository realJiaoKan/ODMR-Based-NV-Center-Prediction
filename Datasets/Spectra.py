from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from settings import RANDOM_SEED

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_PATH = Path("Datasets/Data/data.npz")
# DATA_PATH = Path("Datasets/Data/givenh5.npz")
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
    X = torch.from_numpy(X).float()  # (N, L)
    y = torch.from_numpy(y).float()  # (N, 3)
    full_ds = TensorDataset(X, y)
    train_idx = int(len(full_ds) * (1 - test_ratio))
    test_idx = len(full_ds) - train_idx
    train_ds = torch.utils.data.Subset(full_ds, range(0, train_idx))
    test_ds = torch.utils.data.Subset(full_ds, range(train_idx, len(full_ds)))
    return train_ds, test_ds


def load_loader(shuffle=False, test_ratio=0.2, batch_size=64):
    train_ds, test_ds = load_dataset(shuffle=shuffle, test_ratio=test_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader


def recover_B(B_normalized):
    data = np.load(DATA_PATH)
    reco_params = data["recover_params"]
    Bx = B_normalized[0] * (reco_params[0][1] - reco_params[0][0]) + reco_params[0][0]
    By = B_normalized[1] * (reco_params[1][1] - reco_params[1][0]) + reco_params[1][0]
    Bz = B_normalized[2] * (reco_params[2][1] - reco_params[2][0]) + reco_params[2][0]
    return (Bx, By, Bz)


def plot():
    sample_loader, _ = load_loader(shuffle=True, test_ratio=0.2, batch_size=16)
    X, y = next(iter(sample_loader))
    X, y = X.numpy(), y.numpy()
    fig, axes = plt.subplots(4, 1, figsize=(9, 16))
    for i in range(4):
        Bx, By, Bz = recover_B(y[i])
        axes[i].set_title(
            rf"$\mathbf{{B}}_x={Bx:.2f}, \mathbf{{B}}_y={By:.2f}, \mathbf{{B}}_z={Bz:.2f}$",
            fontsize=16,
        )
        axes[i].plot(X[i])
        axes[i].set_xlabel("Frequency above min (MHz)", fontsize=14)
        axes[i].set_ylabel("Normalized Signal", fontsize=14)
    fig.suptitle(f"X: {X.shape}, y: {y.shape}", y=0.99, fontsize=16)

    plt.tight_layout()
    plt.savefig(SAMPLE_PATH)
    plt.close(fig)


if __name__ == "__main__":
    plot()
