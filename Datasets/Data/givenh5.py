import re
import h5py
import numpy as np

from Datasets.Data.generator import normalize_spectra, param_conversion_normalized

DATASET_PATH = "Reference/dsetsf.h5"
SAMPLE_POINTS = 2000
PREPROCESS_MODE = "invert_norm"


def parse_key(key):
    # "{B, theta, phi}" -> (B, theta, phi)
    vals = list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(key))))
    if len(vals) < 3:
        raise ValueError(f"Cannot parse B, theta, phi from key: {key}")
    return float(vals[0]), float(vals[1]), float(vals[2])


def make_common_grid(freq):
    fmin = min(np.min(f) for f in freq)
    fmax = max(np.max(f) for f in freq)
    print(f"Common grid from {fmin} to {fmax} with {SAMPLE_POINTS} points.")
    return np.linspace(fmin, fmax, SAMPLE_POINTS)


def resample(freq, signal, grid):
    order = np.argsort(freq)
    f_sorted = freq[order]
    s_sorted = signal[order]

    s_interp = np.interp(grid, f_sorted, s_sorted, left=np.nan, right=np.nan)

    if np.isnan(s_interp).any():
        idx = np.where(~np.isnan(s_interp))[0]
        if idx.size == 0:
            return np.full_like(grid, np.nan)
        first, last = idx[0], idx[-1]
        s_interp[:first] = s_interp[first]
        s_interp[last + 1 :] = s_interp[last]

    return s_interp


def load_h5(path):
    param_combs = []
    freq = []
    sig = []
    with h5py.File(path, "r") as f:
        for k in f.keys():
            arr = np.asarray(f[k])
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"Unexpected dataset shape for key {k}: {arr.shape}")
            B, theta, phi = parse_key(k)
            freq_vals, sig_vals = arr[:, 0].astype(float), arr[:, 1].astype(float)
            param_combs.append((B, theta, phi))
            freq.append(freq_vals)
            sig.append(sig_vals)
    grid = make_common_grid(freq)
    spectra = []
    for i in range(len(freq)):
        s_interp = resample(freq[i], sig[i], grid)
        spectra.append(s_interp)
    spectra = np.array(spectra)

    return param_combs, spectra


def param_conversion_normalized(param_combs):
    param_converted = []
    recover_params = []
    for B, theta, phi in param_combs:
        Bx = B * np.sin(theta) * np.cos(phi)
        By = B * np.sin(theta) * np.sin(phi)
        Bz = B * np.cos(theta)
        param_converted.append((Bx, By, Bz))
    param_converted = np.array(param_converted)
    for i in range(3):
        min_val = np.min(param_converted[:, i])
        max_val = np.max(param_converted[:, i])
        param_converted[:, i] = (param_converted[:, i] - min_val) / (max_val - min_val)
        recover_params.append((min_val, max_val))
    return np.array(param_converted), recover_params


if __name__ == "__main__":
    param_combs, spectra = load_h5("Reference/dsetsf.h5")
    X = normalize_spectra(spectra)
    y, recover_params = param_conversion_normalized(param_combs)
    centers = np.array([[0]])
    np.savez_compressed(
        "Datasets/Data/givenh5.npz",
        X=X,
        y=y,
        centers=centers,
        recover_params=recover_params,
    )
    print(
        f"Shape of X: {X.shape}, Shape of y: {y.shape}, Shape of centers: {centers.shape}, Recover parameters: {recover_params}"
    )
    print(f"Sample X: {X[:1]}, y: {y[:1]}, centers: {centers[:1][:3]}")
