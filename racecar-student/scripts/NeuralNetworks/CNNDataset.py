import numpy as np

scans = []

scans.append(scan.copy())

# after collecting data:
scans = np.asarray(scans, dtype=np.float32)

np.save("clean_scans.npy", scans)