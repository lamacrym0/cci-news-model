import numpy as np
import os
import sys
import subprocess

# === CONFIG ===
data_dir = "dummy_data"
os.makedirs(data_dir, exist_ok=True)

n_train = 1000
n_val = 200
n_features = 5

# === Generate dummy data ===
def make_data(n_samples, name):
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    np.save(f"{data_dir}/X_{name}.npy", X)
    np.save(f"{data_dir}/y_{name}.npy", y)

make_data(n_train, "train")
make_data(n_val, "val")

print("✅ Dummy data generated.")


# === Run your pipeline ===
# You can now run your training script like:
# python train.py --data_dir dummy_data --arch lstm --seq_len 30 --batch_size 16
