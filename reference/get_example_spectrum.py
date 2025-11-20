import numpy as np, h5py


def get_example_spectrum(n_samples=5):
    with h5py.File("data/dsetsf.h5", "r") as f:
        keys = list(f.keys())
        sel_keys = keys[:n_samples]
        arrs = [np.asarray(f[k]) for k in sel_keys]

    # Save as an object array to safely handle variable lengths
    out = np.array(arrs, dtype=object)
    np.save("data/example_spectrum.npy", out)
    print(
        f"Saved {len(arrs)} example spectra from first keys:",
        sel_keys,
        "shapes",
        [a.shape for a in arrs],
    )


if __name__ == "__main__":
    get_example_spectrum()
