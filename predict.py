import joblib
import numpy as np
import matplotlib.pyplot as plt

from dataset import preprocess_signal, resample
from settings import *


def predict(spectrum_path: str = None, max_plots: int = 5):
    bundle = joblib.load(MODEL_SAVE_PATH)
    model = bundle["model"]
    grid = bundle["grid"]

    arr = np.load(spectrum_path, allow_pickle=True)
    # Normalize input to a list of 2D arrays with columns (freq, signal)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        spectra = [np.asarray(a) for a in arr]
    else:
        spectra = [np.asarray(arr)]

    X_list = []
    for a in spectra:
        freq, sig = a[:, 0].astype(float), a[:, 1].astype(float)
        xi = resample(freq, sig, grid)
        xi = preprocess_signal(xi, PREPROCESS_MODE)
        xi = np.nan_to_num(xi, nan=0.0).astype(np.float32)
        X_list.append(xi)
    X = np.vstack([x[None, :] for x in X_list])

    y = model.predict(X)

    plt.figure(figsize=(10, 6))
    nplot = min(max_plots, X.shape[0])
    for i in range(nplot):  # overlay first few spectra
        plt.plot(
            grid,
            X[i],
            label=f"Predicted Spectrum {i+1} (B={y[i,0]:.2f}, θ={y[i,1]:.2f}, φ={y[i,2]:.2f})",
        )
    plt.xlabel("Frequency")
    plt.ylabel("Processed Signal")
    plt.title("Sample Spectra Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/spectrum_prediction.png")
    plt.show()


if __name__ == "__main__":
    predict("data/example_spectrum.npy")
