# train.py
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from model import make_model_big as make_model


# ---------------------------
# Hyperparams
# ---------------------------
EPOCHS = 150
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PATIENCE = 15
SEQUENCE_LENGTH = 6

# Pondération de la loss par l'amplitude des cibles (ΔCCI)
ALPHA = 3.0       # importance donnée aux gros |ΔCCI|
MAX_WEIGHT = 5.0  # borne de sécurité des poids
GRAD_CLIP_NORM = 1.0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------
# Prepare model directory
# ---------------------------
base_dir = "models"
os.makedirs(base_dir, exist_ok=True)

existing = [d for d in os.listdir(base_dir) if d.startswith("model_")]
next_id = 0
if existing:
    next_id = max(int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()) + 1

run_dir = os.path.join(base_dir, f"model_{next_id}")
os.makedirs(run_dir, exist_ok=True)
print(f"Training run directory: {run_dir}")


# ---------------------------
# Load tensors
# ---------------------------
X = torch.load("data/X.pt")        # (N, SEQUENCE_LENGTH, n_features)
y = torch.load("data/y.pt")        # (N, 1)
metadata = torch.load("data/metadata.pt")

n_samples, seq_len, n_features = X.shape
assert seq_len == SEQUENCE_LENGTH, f"Expected seq_len={SEQUENCE_LENGTH}, got {seq_len}"

print(f"Loaded X: {X.shape}, y: {y.shape}")
print(f"Total sequences: {n_samples}, Features: {n_features}, Sequence length: {seq_len}")


# ---------------------------
# Chronological split (by sequence index)
# ---------------------------
n_val = int(n_samples * VAL_SPLIT)
n_train = n_samples - n_val

train_X, val_X = X[:n_train], X[n_train:]
train_y, val_y = y[:n_train], y[n_train:]

train_metadata = metadata["metadata"][:n_train]
val_metadata = metadata["metadata"][n_train:]


# ---------------------------
# Normalize features (X) and target (y) using TRAIN stats
# ---------------------------
# X: normalize per feature across time and samples
x_mean = train_X.mean(dim=(0, 1), keepdim=True)  # (1, 1, F)
x_std = train_X.std(dim=(0, 1), keepdim=True) + 1e-6
train_X = (train_X - x_mean) / x_std
val_X = (val_X - x_mean) / x_std

# y: normalize scalar target
y_mean = train_y.mean()
y_std = train_y.std() + 1e-8
train_y = (train_y - y_mean) / y_std
val_y = (val_y - y_mean) / y_std

# Save scalers
torch.save({"mean": x_mean, "std": x_std}, os.path.join(run_dir, "x_scaler.pt"))
torch.save({"mean": y_mean, "std": y_std}, os.path.join(run_dir, "y_scaler.pt"))


# ---------------------------
# DataLoaders
# ---------------------------
train_ds = TensorDataset(train_X, train_y)
val_ds = TensorDataset(val_X, val_y)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ---------------------------
# Model, loss, optim
# ---------------------------
model = make_model(
    input_dim=n_features,
    sequence_length=SEQUENCE_LENGTH,
    hidden_dim=128,
    lstm_layers=2,
    dropout=0.25,
    feature_dropout_p=0.05,
    use_attention=True
).to(DEVICE)

base_criterion = nn.SmoothL1Loss(reduction="none")
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)


def weighted_smooth_l1(pred, target):
    """
    SmoothL1 pondérée : accentue les gros |ΔCCI|.
    Poids = 1 + ALPHA * |target_normalized|, borné.
    """
    base = base_criterion(pred, target)
    weights = (1.0 + ALPHA * torch.abs(target)).clamp_(max=MAX_WEIGHT)
    return (base * weights).mean()


# ---------------------------
# Training loop
# ---------------------------
best_val = float("inf")
patience_counter = 0
log_records = []

print(f"Starting training on {DEVICE}...")
for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)  # (B, 1)
        loss = weighted_smooth_l1(preds, yb)
        loss.backward()
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_dl.dataset)

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = weighted_smooth_l1(preds, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_dl.dataset)

    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    # Log
    log_records.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "lr": lr,
    })
    print(f"Epoch {epoch:03d}/{EPOCHS} | Train {train_loss:.5f} | Val {val_loss:.5f} | LR {lr:.2e}")

    # Early stopping + save best
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
            "hparams": {
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "ALPHA": ALPHA,
                "MAX_WEIGHT": MAX_WEIGHT,
                "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
                "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
            },
            "metadata": {
                "n_train": n_train,
                "n_val": n_val,
                "n_features": n_features,
                "feature_cols": metadata.get("feature_cols", None),
            }
        }, os.path.join(run_dir, "best_model.pt"))
        print(f"  New best model saved (val_loss={val_loss:.5f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break


# ---------------------------
# Save logs & summary
# ---------------------------
df_log = pd.DataFrame(log_records)
df_log.to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

# Optional: save sample predictions on val set
model.eval()
with torch.no_grad():
    sample_preds = model(val_X[:5].to(DEVICE)).cpu()
    sample_true = val_y[:5]
    sample_dates = [m["target_date"] for m in val_metadata[:5]]
    sample_countries = [m["country"] for m in val_metadata[:5]]

summary = f"""
LSTM Training Summary
{'-'*50}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Device: {DEVICE}
Run ID: model_{next_id}
Dataset:
  Total sequences: {n_samples}
  Train / Val: {n_train} /  {n_val}
  Sequence length: {SEQUENCE_LENGTH}
  Features: {n_features}
Training:
  Epochs run: {len(df_log)}
  Best val loss: {best_val:.6f}
  LR: {LR} → {scheduler.get_last_lr()[0]:.2e}
  Batch size: {BATCH_SIZE}
  Weight decay: {WEIGHT_DECAY}
  Alpha (loss weight): {ALPHA}
  Max weight: {MAX_WEIGHT}
Model saved in: {run_dir}
Sample predictions (first 5 val):
"""
for i in range(5):
    summary += f"\n  [{sample_countries[i]}] {sample_dates[i]} | True: {sample_true[i][0]:.4f} | Pred: {sample_preds[i][0]:.4f}"

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(summary.strip())

print(summary)
print(f"Model, scalers, logs & summary saved in: {run_dir}")