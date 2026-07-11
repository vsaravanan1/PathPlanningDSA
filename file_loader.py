import numpy as np
import os

data_dir = "/Users/austinktaylor/projects/PathPlanningDSA/Dataset"

files = os.listdir(data_dir)

npy_files = [f for f in files if f.endswith('.npy')]

example = np.load(os.path.join(data_dir, npy_files[0]))
print(example.shape)

for filename in npy_files:
    file_path = os.path.join(data_dir, filename)
    grid = np.load(file_path)
    print(f"File: {filename}, shape: {grid.shape}")