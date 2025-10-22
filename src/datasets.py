import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl

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


class MLPPolars(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.X_lazy = pl.scan_parquet(x_path)
        self.y_lazy = pl.scan_parquet(y_path)
        # materialize index column once to get lengths and to validate alignment
        x_idx = self.X_lazy.select(pl.col('index')).collect()
        y_idx = self.y_lazy.select(pl.col('index')).collect()
        assert x_idx.equals(y_idx)
        self._n = x_idx.height

    def __len__(self):
        return self._n
    
    def __getitem__(self, idx: int):
        x_df = self.X_lazy.filter(pl.col('index') == idx).collect()
        y_df = self.y_lazy.filter(pl.col('index') == idx).collect()
        if x_df.height == 0 or y_df.height == 0:
            raise IndexError(idx)
        x_tensor = x_df.select(pl.exclude('index')).to_torch(return_type='tensor').float()
        y_tensor = y_df.select(pl.exclude('index')).to_torch(return_type='tensor').float()
        return x_tensor, y_tensor

class LSTMPolars(Dataset):
    def __init__(self, x_path, y_path, sequence_length: int):
        self.X_lazy = pl.scan_parquet(x_path)
        self.y_lazy = pl.scan_parquet(y_path)
        pl.scan_csv
        self.sequence_length  =sequence_length
        # materialize index column once to get lengths and to validate alignment
        x_idx = self.X_lazy.select(pl.col('index')).collect()
        y_idx = self.y_lazy.select(pl.col('index')).collect()
        assert x_idx.frame_equal(y_idx)
        self._n = x_idx.height - self.sequence_length

    def __len__(self):
        return self._n

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.sequence_length  # exclusive
        x_df = self.X_lazy.filter(pl.col('index').is_between(start,end)).collect()
        y_df = self.y_lazy.filter(pl.col('index') == end).collect()
        if x_df.height == 0 or y_df.height == 0:
            raise IndexError(idx)
        x_tensor = x_df.to_torch(return_type='tensor', dtype=torch.float32)
        y_tensor = y_df.to_torch(return_type='tensor', dtype=torch.float32)
        return x_tensor, y_tensor
