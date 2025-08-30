import numpy as np
import matplotlib.pyplot as plt
import h5py

# import h5 file

f = h5py.File("data/dsetsf.h5", "r")
Bpars = [x.replace("{", "").replace("}", "") for x in np.array(list(f.keys()))]
ddic = {}
# make a dictionary: keys are (B, theta, phi), values are (freqs, signal)

for i, key in enumerate(f.keys()):
    ddic[Bpars[i]] = np.array(f[key])
    print(f"key: {Bpars[i]}, shape: {ddic[Bpars[i]].shape}")

print(list(ddic.items())[:1])

f.close()
