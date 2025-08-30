import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import h5py
import numpy as np
import matplotlib.pyplot as plt

from settings import *


@dataclass
class Spectrum:
    B: float
    theta: float
    phi: float
    freq: np.ndarray  # shape (N,)
    signal: np.ndarray  # shape (N,)


def parse_key(key: str) -> Tuple[float, float, float]:
    """Parse dataset key like '{100.645, 0.304242, 5.5785}' -> (B, theta, phi)."""
    vals = list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(key))))
    if len(vals) < 3:
        raise ValueError(f"Cannot parse B, theta, phi from key: {key}")
    return float(vals[0]), float(vals[1]), float(vals[2])


def load_h5(path: str) -> List[Spectrum]:
    """
    Load all spectra from the provided h5 file.

    Each dataset is expected to be a 2D array of shape (N, 2) where column 0 is
    frequency and column 1 is the signal.
    """
    specs: List[Spectrum] = []
    with h5py.File(path, "r") as f:
        for k in f.keys():
            arr = np.asarray(f[k])
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"Unexpected dataset shape for key {k}: {arr.shape}")
            B, th, ph = parse_key(k)
            freq, sig = arr[:, 0].astype(float), arr[:, 1].astype(float)
            specs.append(Spectrum(B, th, ph, freq, sig))
    return specs


def make_common_grid(
    specs: Iterable[Spectrum], n_points: int = SAMPLE_POINTS
) -> np.ndarray:
    """Create a common frequency grid covering the entire dataset."""
    fmin = min(np.min(s.freq) for s in specs)
    fmax = max(np.max(s.freq) for s in specs)
    return np.linspace(fmin, fmax, int(n_points))


def resample(
    freq: np.ndarray, signal: np.ndarray, grid: np.ndarray, fill_value: float = np.nan
) -> np.ndarray:
    """
    Interpolate the spectrum onto a common grid using linear interpolation.

    Extrapolation is filled with `fill_value` (defaults to NaN), then NaNs are
    replaced with edge values to avoid artifacts.
    """
    # Ensure monotonic increasing frequencies for interpolation
    order = np.argsort(freq)
    f_sorted = freq[order]
    s_sorted = signal[order]

    # Clip grid to data range for interpolation, then pad with edges
    s_interp = np.interp(grid, f_sorted, s_sorted, left=np.nan, right=np.nan)

    # Replace NaNs by nearest valid edge values
    if np.isnan(s_interp).any():
        # forward fill
        idx = np.where(~np.isnan(s_interp))[0]
        if idx.size == 0:
            return np.full_like(grid, np.nan)
        first, last = idx[0], idx[-1]
        s_interp[:first] = s_interp[first]
        s_interp[last + 1 :] = s_interp[last]
    return s_interp


def preprocess_signal(sig: np.ndarray, mode: str = PREPROCESS_MODE) -> np.ndarray:
    """
    Basic amplitude preprocessing.

    - invert_norm: use 1 - sig then min-max normalize to [0, 1]
    - zscore: standardize to zero mean, unit variance
    - none: return as-is
    """
    if mode == "invert_norm":
        x = 1.0 - sig
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-12)
    if mode == "zscore":
        mu, std = np.nanmean(sig), np.nanstd(sig)
        return (sig - mu) / (std + 1e-12)
    return sig


def build_dataset(
    path: str = DATASET_PATH,
    n_points: int = SAMPLE_POINTS,
    preprocess: str = PREPROCESS_MODE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, y, grid).

    X shape: (n_samples, n_points), y shape: (n_samples, 3)
    """
    specs = load_h5(path)
    grid = make_common_grid(specs, n_points)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for s in specs:
        s_interp = resample(s.freq, s.signal, grid)
        s_proc = preprocess_signal(s_interp, preprocess)
        # replace any lingering NaNs from interpolation with zeros
        s_proc = np.nan_to_num(s_proc, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(s_proc.astype(np.float32))
        y_list.append(np.array([s.B, s.theta, s.phi], dtype=np.float32))

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y, grid


def sample():
    X, y, grid = build_dataset()

    plt.figure(figsize=(10, 6))
    for i in range(min(5, X.shape[0])):  # plot up to 5 spectra
        plt.plot(
            grid,
            X[i],
            label=f"Spectrum {i+1} (B={y[i,0]:.2f}, θ={y[i,1]:.2f}, φ={y[i,2]:.2f})",
        )
    plt.xlabel("Frequency")
    plt.ylabel("Processed Signal")
    plt.title("Sample Spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/dataset_sample.png")
    plt.show()


if __name__ == "__main__":
    sample()
