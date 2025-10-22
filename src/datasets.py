import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------
# Dataset and DataLoader
# ---------------------
class MLPDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        assert len(self.X) == len(self.y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class LSTMDataset(Dataset):
    def __init__(self, x_path, y_path, sequence_length: int):
        self.X = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.sequence_length = sequence_length
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

