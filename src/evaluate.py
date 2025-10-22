# test.py
import argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

import polars as pl

from datasets import MLPPolars, LSTMPolars
from model import MLP, LSTM

def load_checkpoint(fname, model, device):
    ckpt = torch.load(fname, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt

def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
            ys.append(yb.numpy())
    y_true = np.concatenate(ys, axis=0).reshape(-1,1)
    y_pred = np.concatenate(preds, axis=0).reshape(-1,1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse, y_true, y_pred

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/rows")
parser.add_argument("--checkpoint", type=str, required=True, help="path to best checkpoint .pt")
parser.add_argument("--arch", choices=["mlp","lstm"], default="mlp")
parser.add_argument("--seq_len", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--save_preds", type=str, default="", help="optional path to save predictions as npz")
args = parser.parse_args()

device = torch.device(args.device)

# build dataset
data_dir = Path(args.data_dir)
if args.arch == "mlp":
    test_ds = MLPPolars(data_dir / "X.parquet", data_dir / "y.parquet")
else:
    test_ds = LSTMPolars(data_dir / "X.parquet", data_dir / "y.parquet", sequence_length=args.seq_len)

test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# infer shape from sample to build model
sample_x, _ = test_ds[0]
if args.arch == "mlp":
    in_dim = sample_x.numel()
    model = MLP(in_dim, out_dim=1)
else:
    seq_len, feat_dim = sample_x.shape
    model = LSTM(input_size=feat_dim, output_size=1)

model.to(device)
ckpt = load_checkpoint(args.checkpoint, model, device)

rmse, y_true, y_pred = evaluate(model, test_loader, device)

print(f"Test RMSE: {rmse:.5f}")
if args.save_preds:
    out_path = Path(args.save_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # build a tidy DataFrame and save as parquet
    df = pl.DataFrame({
        "y_true": y_true.reshape(-1),
        "y_pred": y_pred.reshape(-1)
    })
    # prefer pyarrow engine if available
    df.write_parquet(out_path)
    print(f"Saved predictions to {out_path}")
