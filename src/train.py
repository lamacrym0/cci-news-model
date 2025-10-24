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

from model import make_model



# Hyperparameters

EPOCHS = 150
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PATIENCE = 15
SEQUENCE_LENGTH = 60

# Weighted loss
ALPHA = 3.0
MAX_WEIGHT = 5.0
GRAD_CLIP_NORM = 1.0

# Model architecture
HIDDEN_DIM = 512
LSTM_LAYERS = 6
ATTN_HEADS = 8
DROPOUT = 0.3
FEATURE_DROPOUT = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)


# Prepare run directory

base_dir = "models"
os.makedirs(base_dir, exist_ok=True)

existing = [d for d in os.listdir(base_dir) if d.startswith("model_")]
next_id = max([int(d.split("_")[1]) for d in existing], default=-1) + 1

run_dir = os.path.join(base_dir, f"model_{next_id}")
os.makedirs(run_dir, exist_ok=True)
print(f"Training run directory: {run_dir}")


# Load data

X = torch.load("data/X_us.pt")  # (N, seq_len, n_features)
y = torch.load("data/y_us.pt")  # (N, 1)

n_samples, seq_len, n_features = X.shape
assert seq_len == SEQUENCE_LENGTH, f"Expected seq_len={SEQUENCE_LENGTH}, got {seq_len}"

print(f"Loaded X: {X.shape}, y: {y.shape}")
print(f"Total sequences: {n_samples}, Features: {n_features}, Sequence length: {seq_len}")
print("Note: Raw data (not normalized)")

# --- split chronologique ---
n_val   = int(n_samples * VAL_SPLIT)
n_train = n_samples - n_val

train_X_raw, val_X_raw = X[:n_train], X[n_train:]
train_y_raw, val_y_raw = y[:n_train], y[n_train:]

# --- normalisation ---
X_mean = train_X_raw.mean(dim=(0, 1), keepdim=True)
X_std  = train_X_raw.std(dim=(0, 1), keepdim=True) + 1e-8
y_mean = train_y_raw.mean()
y_std  = train_y_raw.std() + 1e-8

train_X = (train_X_raw - X_mean) / X_std
val_X   = (val_X_raw   - X_mean) / X_std

train_y = (train_y_raw - y_mean) / y_std
val_y   = (val_y_raw   - y_mean) / y_std

# sauvegarde des scalers (train-only)
torch.save({"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std},
           os.path.join(run_dir, "scalers.pt"))



# DataLoaders

train_ds = TensorDataset(train_X, train_y)
val_ds = TensorDataset(val_X, val_y)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# Model

model = make_model(
    input_dim=n_features,
    sequence_length=SEQUENCE_LENGTH,
    hidden_dim=HIDDEN_DIM,
    lstm_layers=LSTM_LAYERS,
    attn_heads=ATTN_HEADS,
    dropout=DROPOUT,
    feature_dropout_p=FEATURE_DROPOUT
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")


# Loss / Optimizer / Scheduler

base_criterion = nn.SmoothL1Loss(reduction="none")

def weighted_smooth_l1(pred, target):
    """
    SmoothL1 pondérée : accentue les gros |ΔCCI|.
    Poids = 1 + ALPHA * |target_normalized|, borné à MAX_WEIGHT.
    """
    base = base_criterion(pred, target)
    weights = (1.0 + ALPHA * torch.abs(target)).clamp_(max=MAX_WEIGHT)
    return (base * weights).mean()

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)


# Training Loop

best_val = float("inf")
patience_counter = 0
log_records = []

print(f"\nStarting training on {DEVICE}...")
print(f"Architecture: {LSTM_LAYERS} LSTM layers, hidden_dim={HIDDEN_DIM}, {ATTN_HEADS} attention heads")
print(f"Loss: Weighted SmoothL1 (alpha={ALPHA}, max_weight={MAX_WEIGHT})")

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
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
    log_records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
    print(f"Epoch {epoch:03d}/{EPOCHS} | Train {train_loss:.5f} | Val {val_loss:.5f} | LR {lr:.2e}")

    # --- Early stopping & save best ---
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
                "HIDDEN_DIM": HIDDEN_DIM,
                "LSTM_LAYERS": LSTM_LAYERS,
                "ATTN_HEADS": ATTN_HEADS,
                "DROPOUT": DROPOUT,
                "FEATURE_DROPOUT": FEATURE_DROPOUT,
            },
            "metadata": {
                "n_train": n_train,
                "n_val": n_val,
                "n_features": n_features,
                "n_params": n_params,
            }
        }, os.path.join(run_dir, "best_model.pt"))
        print(f"  ✓ New best model saved (val_loss={val_loss:.5f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break


# Save logs

df_log = pd.DataFrame(log_records)
df_log.to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

# Plot loss curve
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(df_log["epoch"], df_log["train_loss"], label="Train")
    plt.plot(df_log["epoch"], df_log["val_loss"], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"⚠️ Could not plot loss curve: {e}")


# Summary

summary = f"""
LSTM Training Summary
{'-'*60}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Device: {DEVICE}
Run ID: model_{next_id}

Dataset:
  Total sequences: {n_samples}
  Train / Val: {n_train} / {n_val} ({VAL_SPLIT*100:.0f}% val split)
  Sequence length: {SEQUENCE_LENGTH} days
  Features: {n_features}

Model Architecture:
  Type: Deep Bidirectional LSTM + Multi-head Attention
  LSTM layers: {LSTM_LAYERS}
  Hidden dimension: {HIDDEN_DIM}
  Attention heads: {ATTN_HEADS}
  Total parameters: {n_params:,}
  Dropout: {DROPOUT}
  Feature dropout: {FEATURE_DROPOUT}

Training:
  Epochs run: {len(df_log)}
  Best val loss: {best_val:.6f}
  Initial LR: {LR} → Final LR: {scheduler.get_last_lr()[0]:.2e}
  Batch size: {BATCH_SIZE}
  Weight decay: {WEIGHT_DECAY}
  Loss: Weighted SmoothL1 (alpha={ALPHA}, max_weight={MAX_WEIGHT})
  Grad clip: {GRAD_CLIP_NORM}

Outputs saved in: {run_dir}
"""

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(summary.strip())

print("\n" + summary)
print(f"✅ Model, scalers, logs & summary saved in: {run_dir}")
