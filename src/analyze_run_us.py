# evaluate_model.py
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import make_model_big


# ======================================================
# ⚙️ Configuration
# ======================================================
MODEL_DIR = "model/model_4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 📦 Load model + scalers + logs
# ======================================================
ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
scalers_path = os.path.join(MODEL_DIR, "scalers.pt")
log_path = os.path.join(MODEL_DIR, "train_log.csv")

assert os.path.exists(ckpt_path), f"❌ No checkpoint found at {ckpt_path}"

checkpoint = torch.load(ckpt_path, map_location=DEVICE)
hparams = checkpoint["hparams"]

print(f"✅ Loaded model from {ckpt_path}")
print(f"Trained for {checkpoint['epoch']} epochs, best val loss = {checkpoint['val_loss']:.5f}")

# Load scalers
scalers = torch.load(scalers_path)
X_mean, X_std = scalers["X_mean"], scalers["X_std"]
y_mean, y_std = scalers["y_mean"], scalers["y_std"]

# Load logs
df_log = pd.read_csv(log_path)

# ======================================================
# 📈 Plot training & validation loss
# ======================================================
plt.figure(figsize=(8,5))
plt.plot(df_log["epoch"], df_log["train_loss"], label="Train Loss", lw=2)
plt.plot(df_log["epoch"], df_log["val_loss"], label="Val Loss", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "01_loss_curves.png"))
plt.close()
print("📊 Saved: 01_loss_curves.png")

# ======================================================
# 🧠 Reload model architecture
# ======================================================
model = make_model_big(
    input_dim=checkpoint["metadata"]["n_features"],
    sequence_length=hparams["SEQUENCE_LENGTH"],
    hidden_dim=hparams["HIDDEN_DIM"],
    lstm_layers=hparams["LSTM_LAYERS"],
    attn_heads=hparams["ATTN_HEADS"],
    dropout=hparams["DROPOUT"],
    feature_dropout_p=hparams["FEATURE_DROPOUT"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state"])
model.eval()

# ======================================================
# 📊 Load data (same tensors as training)
# ======================================================
X = torch.load("data/X_us.pt")
y = torch.load("data/y_us.pt")

SEQ_LEN = hparams["SEQUENCE_LENGTH"]
assert X.shape[1] == SEQ_LEN

# Normalize using saved scalers
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# Split chronologically (same as train)
n_samples = len(X)
n_val = int(n_samples * 0.2)
n_train = n_samples - n_val
val_X, val_y = X_norm[n_train:], y_norm[n_train:]

print(f"Validation samples: {len(val_X)}")

# ======================================================
# 🔮 Make predictions
# ======================================================
with torch.no_grad():
    preds_norm = model(val_X.to(DEVICE)).cpu()

# Denormalize
preds = preds_norm * y_std + y_mean
true = val_y * y_std + y_mean

# Convert to numpy
preds_np = preds.numpy().flatten()
true_np = true.numpy().flatten()

# ======================================================
# 📈 Scatter: true vs predicted
# ======================================================
plt.figure(figsize=(6,6))
plt.scatter(true_np, preds_np, alpha=0.5, edgecolor="k", s=20)
lims = [min(true_np.min(), preds_np.min()), max(true_np.max(), preds_np.max())]
plt.plot(lims, lims, "r--", label="Ideal")
plt.xlabel("True ΔCCI")
plt.ylabel("Predicted ΔCCI")
plt.title("True vs Predicted ΔCCI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "02_predictions_scatter.png"))
plt.close()
print("📊 Saved: 02_predictions_scatter.png")

# ======================================================
# 📉 Residuals plot
# ======================================================
residuals = preds_np - true_np

plt.figure(figsize=(8,5))
plt.scatter(true_np, residuals, alpha=0.5, edgecolor="k", s=20)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("True ΔCCI")
plt.ylabel("Residuals (Pred - True)")
plt.title("Residuals vs True Values")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "03_residuals_analysis.png"))
plt.close()
print("📊 Saved: 03_residuals_analysis.png")

# ======================================================
# 📊 Error by magnitude
# ======================================================
abs_error = np.abs(residuals)
bins = np.linspace(true_np.min(), true_np.max(), 20)
digitized = np.digitize(true_np, bins)
mean_err = [abs_error[digitized == i].mean() for i in range(1, len(bins))]

plt.figure(figsize=(8,5))
plt.bar(bins[:-1], mean_err, width=np.diff(bins), align="edge", edgecolor="black")
plt.xlabel("True ΔCCI bins")
plt.ylabel("Mean Absolute Error")
plt.title("MAE by ΔCCI magnitude")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "04_error_by_magnitude.png"))
plt.close()
print("📊 Saved: 04_error_by_magnitude.png")

# ======================================================
# 📆 Time series samples (first 200 val)
# ======================================================
plt.figure(figsize=(12,5))
plt.plot(true_np[:200], label="True", lw=2)
plt.plot(preds_np[:200], label="Predicted", lw=2)
plt.xlabel("Time index (chronological val set)")
plt.ylabel("ΔCCI (denormalized)")
plt.title("ΔCCI - Time Series (first 200 val samples)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "05_time_series_samples.png"))
plt.close()
print("📊 Saved: 05_time_series_samples.png")

# ======================================================
# 📊 Error distribution
# ======================================================
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=40, color="skyblue", edgecolor="black")
plt.axvline(0, color="r", linestyle="--")
plt.xlabel("Residuals (Pred - True)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "06_residual_distribution.png"))
plt.close()
print("📊 Saved: 06_residual_distribution.png")

# ======================================================
# 🧮 Metrics summary
# ======================================================
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))
corr = np.corrcoef(preds_np, true_np)[0,1]

summary = f"""
Evaluation Summary
{'-'*50}
Model directory: {MODEL_DIR}
Device: {DEVICE}

Metrics:
  MAE   = {mae:.6f}
  RMSE  = {rmse:.6f}
  Corr  = {corr:.4f}
  Samples (val): {len(val_X)}

Plots saved:
  01_loss_curves.png
  02_predictions_scatter.png
  03_residuals_analysis.png
  04_error_by_magnitude.png
  05_time_series_samples.png
  06_residual_distribution.png
"""

with open(os.path.join(MODEL_DIR, "evaluation_summary.txt"), "w") as f:
    f.write(summary.strip())

print("\n" + summary)
