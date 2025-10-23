# evaluate.py
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from model import make_model_big

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ---------------------------
# Configuration
# ---------------------------
MODEL_DIR = "model/model_3"  # Changez selon votre modèle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512

if len(sys.argv) > 1:
    MODEL_DIR = sys.argv[1]

if not os.path.exists(MODEL_DIR):
    print(f"❌ Model directory not found: {MODEL_DIR}")
    print("Usage: python evaluate.py [model_dir]")
    sys.exit(1)

print(f"📊 Evaluating model: {MODEL_DIR}")
print(f"Device: {DEVICE}\n")

# ---------------------------
# Load data and metadata
# ---------------------------
X = torch.load("data/X.pt")
y = torch.load("data/y.pt")
metadata = torch.load("data/metadata.pt")

# Extract normalization stats
X_mean = metadata["X_mean"]
X_std = metadata["X_std"]
y_mean = metadata["y_mean"]
y_std = metadata["y_std"]
sequence_metadata = metadata["metadata"]

n_samples = len(X)
n_features = X.shape[-1]

print(f"Dataset: {n_samples} sequences, {n_features} features")
print(f"y_mean={y_mean:.4f}, y_std={y_std:.4f}\n")

# ---------------------------
# Load model checkpoint
# ---------------------------
checkpoint = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
hparams = checkpoint["hparams"]
model_metadata = checkpoint["metadata"]

print("Model hyperparameters:")
for k, v in hparams.items():
    print(f"  {k}: {v}")
print(f"\nModel info:")
print(f"  n_params: {model_metadata['n_params']:,}")
print(f"  n_train: {model_metadata['n_train']}")
print(f"  n_val: {model_metadata['n_val']}")
print(f"  Best val_loss: {checkpoint['val_loss']:.6f}")
print(f"  Epoch: {checkpoint['epoch']}\n")

# Reconstruct model
model = make_model_big(
    input_dim=n_features,
    sequence_length=hparams["SEQUENCE_LENGTH"],
    hidden_dim=hparams["HIDDEN_DIM"],
    lstm_layers=hparams["LSTM_LAYERS"],
    attn_heads=hparams["ATTN_HEADS"],
    dropout=hparams["DROPOUT"],
    feature_dropout_p=hparams["FEATURE_DROPOUT"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state"])
model.eval()
print("✅ Model loaded successfully\n")

# ---------------------------
# Split data (same split as training)
# ---------------------------
n_train = model_metadata["n_train"]
n_val = model_metadata["n_val"]

train_X, val_X = X[:n_train], X[n_train:]
train_y, val_y = y[:n_train], y[n_train:]
train_meta = sequence_metadata[:n_train]
val_meta = sequence_metadata[n_train:]

# ---------------------------
# Inference on train and val
# ---------------------------
def get_predictions(X_data, y_data):
    """Get predictions for a dataset"""
    dataset = TensorDataset(X_data, y_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu()
            all_preds.append(preds)
            all_targets.append(yb)
    
    return torch.cat(all_preds), torch.cat(all_targets)

print("🔮 Running inference...")
train_preds, train_targets = get_predictions(train_X, train_y)
val_preds, val_targets = get_predictions(val_X, val_y)

# Denormalize
train_preds_denorm = train_preds * y_std + y_mean
train_targets_denorm = train_targets * y_std + y_mean
val_preds_denorm = val_preds * y_std + y_mean
val_targets_denorm = val_targets * y_std + y_mean

print("✅ Inference complete\n")

# ---------------------------
# Compute metrics
# ---------------------------
def compute_metrics(y_true, y_pred, name=""):
    """Compute regression metrics"""
    y_true_np = y_true.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()
    
    mae = mean_absolute_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    r2 = r2_score(y_true_np, y_pred_np)
    
    # Direction accuracy (sign match)
    direction_acc = (np.sign(y_true_np) == np.sign(y_pred_np)).mean()
    
    print(f"{name} Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Direction accuracy: {direction_acc*100:.2f}%")
    print()
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "direction_acc": direction_acc}

train_metrics = compute_metrics(train_targets_denorm, train_preds_denorm, "TRAIN")
val_metrics = compute_metrics(val_targets_denorm, val_preds_denorm, "VAL")

# ---------------------------
# Create output directory
# ---------------------------
output_dir = os.path.join(MODEL_DIR, "evaluation")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Plot 1: Training curves
# ---------------------------
if os.path.exists(os.path.join(MODEL_DIR, "train_log.csv")):
    df_log = pd.read_csv(os.path.join(MODEL_DIR, "train_log.csv"))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(df_log["epoch"], df_log["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(df_log["epoch"], df_log["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(df_log["epoch"], df_log["lr"], color="green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_training_curves.png"), dpi=150, bbox_inches="tight")
    print("✅ Saved: 01_training_curves.png")
    plt.close()

# ---------------------------
# Plot 2: Predictions vs Actuals (scatter)
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, preds, targets, title, metrics in zip(
    axes, 
    [train_preds_denorm, val_preds_denorm],
    [train_targets_denorm, val_targets_denorm],
    ["Train Set", "Validation Set"],
    [train_metrics, val_metrics]
):
    preds_np = preds.numpy().flatten()
    targets_np = targets.numpy().flatten()
    
    ax.scatter(targets_np, preds_np, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(targets_np.min(), preds_np.min())
    max_val = max(targets_np.max(), preds_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect prediction")
    
    ax.set_xlabel("Actual ΔCCI")
    ax.set_ylabel("Predicted ΔCCI")
    ax.set_title(f"{title}\nR²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_predictions_scatter.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: 02_predictions_scatter.png")
plt.close()

# ---------------------------
# Plot 3: Residuals analysis
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (preds, targets, title) in enumerate(zip(
    [train_preds_denorm, val_preds_denorm],
    [train_targets_denorm, val_targets_denorm],
    ["Train", "Val"]
)):
    preds_np = preds.numpy().flatten()
    targets_np = targets.numpy().flatten()
    residuals = targets_np - preds_np
    
    row = idx
    
    # Residuals vs predictions
    axes[row, 0].scatter(preds_np, residuals, alpha=0.3, s=10)
    axes[row, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[row, 0].set_xlabel("Predicted ΔCCI")
    axes[row, 0].set_ylabel("Residuals")
    axes[row, 0].set_title(f"{title} - Residuals vs Predictions")
    axes[row, 0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[row, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[row, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[row, 1].set_xlabel("Residuals")
    axes[row, 1].set_ylabel("Frequency")
    axes[row, 1].set_title(f"{title} - Residuals Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}")
    axes[row, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_residuals_analysis.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: 03_residuals_analysis.png")
plt.close()

# ---------------------------
# Plot 4: Error by magnitude
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, preds, targets, title in zip(
    axes,
    [train_preds_denorm, val_preds_denorm],
    [train_targets_denorm, val_targets_denorm],
    ["Train", "Val"]
):
    preds_np = preds.numpy().flatten()
    targets_np = targets.numpy().flatten()
    errors = np.abs(targets_np - preds_np)
    magnitudes = np.abs(targets_np)
    
    ax.scatter(magnitudes, errors, alpha=0.3, s=10)
    ax.set_xlabel("|Actual ΔCCI|")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"{title} - Error vs Target Magnitude")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_error_by_magnitude.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: 04_error_by_magnitude.png")
plt.close()

# ---------------------------
# Plot 5: Time series samples
# ---------------------------
# Select 5 random countries from validation set
unique_countries = list(set([m["country"] for m in val_meta]))
sample_countries = np.random.choice(unique_countries, size=min(5, len(unique_countries)), replace=False)

fig, axes = plt.subplots(5, 1, figsize=(14, 12))

for idx, country in enumerate(sample_countries):
    # Get all sequences for this country
    country_indices = [i for i, m in enumerate(val_meta) if m["country"] == country]
    
    if len(country_indices) == 0:
        continue
    
    dates = [val_meta[i]["target_date"] for i in country_indices]
    actuals = [val_targets_denorm[i].item() for i in country_indices]
    preds = [val_preds_denorm[i].item() for i in country_indices]
    
    axes[idx].plot(dates, actuals, 'o-', label="Actual", linewidth=2, markersize=4)
    axes[idx].plot(dates, preds, 's-', label="Predicted", linewidth=2, markersize=4)
    axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[idx].set_ylabel("ΔCCI")
    axes[idx].set_title(f"{country}")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=45)

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_time_series_samples.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: 05_time_series_samples.png")
plt.close()

# ---------------------------
# Plot 6: Performance by country
# ---------------------------
# Compute MAE by country for validation set
country_maes = {}
for country in unique_countries:
    country_indices = [i for i, m in enumerate(val_meta) if m["country"] == country]
    if len(country_indices) > 0:
        country_actuals = val_targets_denorm[country_indices].numpy().flatten()
        country_preds = val_preds_denorm[country_indices].numpy().flatten()
        country_maes[country] = mean_absolute_error(country_actuals, country_preds)

# Sort by MAE
sorted_countries = sorted(country_maes.items(), key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(12, 8))
countries = [c[0] for c in sorted_countries]
maes = [c[1] for c in sorted_countries]

ax.barh(countries, maes, color='steelblue')
ax.set_xlabel("MAE (ΔCCI)")
ax.set_title("Validation MAE by Country")
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "06_mae_by_country.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: 06_mae_by_country.png")
plt.close()

# ---------------------------
# Save detailed results
# ---------------------------
# Validation predictions with metadata
val_results = []
for i in range(len(val_preds_denorm)):
    val_results.append({
        "country": val_meta[i]["country"],
        "end_date": val_meta[i]["end_date"],
        "target_date": val_meta[i]["target_date"],
        "actual": val_targets_denorm[i].item(),
        "predicted": val_preds_denorm[i].item(),
        "error": val_targets_denorm[i].item() - val_preds_denorm[i].item(),
        "abs_error": abs(val_targets_denorm[i].item() - val_preds_denorm[i].item()),
    })

df_val_results = pd.DataFrame(val_results)
df_val_results.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)
print("✅ Saved: val_predictions.csv")

# Summary statistics
summary = f"""
{'='*70}
MODEL EVALUATION SUMMARY
{'='*70}
Model: {MODEL_DIR}
Device: {DEVICE}

DATASET INFO
------------
Total sequences: {n_samples}
Train: {n_train} | Val: {n_val}
Features: {n_features}

TRAIN METRICS
-------------
MAE:  {train_metrics['mae']:.4f}
RMSE: {train_metrics['rmse']:.4f}
R²:   {train_metrics['r2']:.4f}
Direction accuracy: {train_metrics['direction_acc']*100:.2f}%

VALIDATION METRICS
------------------
MAE:  {val_metrics['mae']:.4f}
RMSE: {val_metrics['rmse']:.4f}
R²:   {val_metrics['r2']:.4f}
Direction accuracy: {val_metrics['direction_acc']*100:.2f}%

TOP 5 COUNTRIES (Best MAE)
--------------------------
"""

for country, mae in sorted_countries[:5]:
    summary += f"{country:8s}: {mae:.4f}\n"

summary += "\nBOTTOM 5 COUNTRIES (Worst MAE)\n"
summary += "--------------------------\n"
for country, mae in sorted_countries[-5:]:
    summary += f"{country:8s}: {mae:.4f}\n"

summary += f"\n{'='*70}\n"
summary += f"All outputs saved in: {output_dir}\n"
summary += f"{'='*70}\n"

with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
    f.write(summary)

print(summary)
print(f"🎉 Evaluation complete! Check {output_dir}/ for all outputs.")