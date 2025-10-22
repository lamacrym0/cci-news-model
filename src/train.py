"""
train.py
Minimal PyTorch training loop with validation, best-checkpoint saving,
and early stopping. Designed for tabular X, y stored as numpy arrays.
"""

import os
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.metrics import mean_squared_error

from datasets import MLPPolars, LSTMPolars
from model import MLP, LSTM

# ---------------------
# Config and arguments
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/processed", help="folder with X_train.npy etc.")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=10, help="early stopping patience (val epochs)")
parser.add_argument("--min_delta", type=float, default=1e-4, help="min improvement to reset patience")
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "lstm"], help="model architecture")
parser.add_argument("--seq_len", type=int, default=30, help="sequence length for LSTM as day")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
torch.device(args.device)

if args.arch == "mlp":
    train_ds = MLPPolars(Path(args.data_dir) / "X.parquet", Path(args.data_dir) / "y.parquet")
    val_ds   = MLPPolars(Path(args.data_dir) / "X.parquet",   Path(args.data_dir) / "y.parquet")
elif args.arch == "lstm":
    train_ds = LSTMPolars(Path(args.data_dir) / "X.parquet", Path(args.data_dir) / "y.parquet", sequence_length=args.seq_len)
    val_ds   = LSTMPolars(Path(args.data_dir) / "X.parquet",   Path(args.data_dir) / "y.parquet",sequence_length=args.seq_len)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)


# infer input dim from dataset
sample_x, _ = train_ds[0]   # sample_x is a tensor
if args.arch == "mlp":
    in_dim = sample_x.numel()
    model = MLP(in_dim, out_dim=1)
elif args.arch == "lstm":
    # sample_x shape: (seq_len, features)
    seq_len, feat_dim = sample_x.shape
    model = LSTM(input_size=feat_dim, output_size=1)
model.to(args.device)


# ---------------------
# Loss, optimizer, scheduler
# ---------------------
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# ---------------------
# Utilities: save / load checkpoint
# ---------------------
def save_checkpoint(state, fname):
    torch.save(state, fname)

def load_checkpoint(fname, model, optimizer=None, scheduler=None):
    ckpt = torch.load(fname, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt.get("epoch", -1), ckpt.get("best_val", None)

# ---------------------
# Training & validation loops
# ---------------------
def validate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
            ys.append(yb.cpu().numpy())
    y_true = np.concatenate(ys, axis=0).reshape(-1,1)
    y_pred = np.concatenate(preds, axis=0).reshape(-1,1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

best_val = float("inf")
best_epoch = -1
no_improve = 0

start_time = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(args.device), yb.to(args.device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_loss = float(np.mean(epoch_losses))

    val_rmse = validate(model, val_loader, args.device)

    # scheduler step requires the metric to reduce on plateau
    scheduler.step(val_rmse)

    # logging (simple)
    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.5f} | val_rmse {val_rmse:.5f} | time {elapsed:.1f}s")

    # early stopping & checkpoint best
    improved = (best_val - val_rmse) > args.min_delta
    if improved:
        best_val = val_rmse
        best_epoch = epoch
        no_improve = 0
        ckpt_path = Path(args.save_dir) / f"best_epoch_{epoch:03d}_val_{best_val:.5f}.pt"
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val": best_val
        }, str(ckpt_path))
        print(f"  Saved best checkpoint to {ckpt_path}")
    else:
        no_improve += 1

    if no_improve >= args.patience:
        print(f"Early stopping triggered. No improvement for {no_improve} epochs. Best val {best_val:.5f} at epoch {best_epoch}.")
        break

# final message, best model path
print(f"Training finished. Best val_rmse {best_val:.5f} at epoch {best_epoch}.")