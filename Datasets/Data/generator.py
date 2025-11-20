import numpy as np

from Datasets.Data.NVODMRSpectrum import gen_spectrum_nv

B_list = np.linspace(50, 160, 1000)
theta_list = np.linspace(0, 180, 1000)
phi_list = np.linspace(0, 360, 1000)

width, amp = 2.25, 0.1
freq_bins = np.linspace(2500.0, 3200.0, 2001)

filename = "Datasets/Data/data.npz"


def shuffle_parameters():
    para_combs = []
    B_perm = np.random.permutation(len(B_list))
    theta_perm = np.random.permutation(len(theta_list))
    phi_perm = np.random.permutation(len(phi_list))
    for i in range(len(B_list)):
        para_combs.append(
            (B_list[B_perm[i]], theta_list[theta_perm[i]], phi_list[phi_perm[i]])
        )
    return para_combs


def generate_spectra(param_combs):
    spectra = []
    centers = []
    for B, theta, phi in param_combs:
        spectrum, center = gen_spectrum_nv(B, theta, phi, width, amp, freq_bins)
        spectra.append(spectrum)
        centers.append(center)
    return np.array(spectra), np.array(centers)


def normalize_spectra(spectra):
    # Normailize based on all spectra to [0, 1]
    min_val = np.min(spectra)
    max_val = np.max(spectra)
    spectra_normed = []
    for spec in spectra:
        norm_spec = 1 - (spec - min_val) / (max_val - min_val)
        spectra_normed.append(norm_spec)
    return np.array(spectra_normed)


if __name__ == "__main__":
    param_combs = shuffle_parameters()
    spectra, centers = generate_spectra(param_combs)
    X = normalize_spectra(spectra)
    y = np.array(param_combs)
    centers = np.array(centers)
    np.savez_compressed(filename, X=X, y=y, centers=centers)
    print(
        f"Shape of X: {X.shape}, Shape of y: {y.shape}, Shape of centers: {centers.shape}"
    )
    print(f"Sample X: {X[:1]}, y: {y[:1]}, centers: {centers[:1]}")
