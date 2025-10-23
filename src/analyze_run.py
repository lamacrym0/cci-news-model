import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from model import make_model


def analyze_run(run_dir: str, X_path="data/X.pt", y_path="data/y.pt", show=True):
    """
    Analyse et visualise les résultats d'un run d'entraînement.
    Affiche :
      - résumé du training
      - courbes de perte
      - distribution des erreurs
      - scatter y_true vs y_pred
      - métriques (MAE, RMSE, R²)
    """
    print(f"\n📁 Analyzing run: {run_dir}")
    log_path = os.path.join(run_dir, "train_log.csv")
    summary_path = os.path.join(run_dir, "summary.txt")
    model_path = os.path.join(run_dir, "best_model.pt")
    x_scaler_path = os.path.join(run_dir, "x_scaler.pt")
    y_scaler_path = os.path.join(run_dir, "y_scaler.pt")

    # ---------------------------
    # 1️⃣ Read summary and logs
    # ---------------------------
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            print("\n📄 Summary:\n" + f.read())
    else:
        print("⚠️ No summary.txt found.")

    if not os.path.exists(log_path):
        print("❌ train_log.csv not found.")
        return

    df_log = pd.read_csv(log_path)
    print("\n📊 Training stats:")
    print(df_log.describe()[["train_loss", "val_loss"]])

    # ---------------------------
    # 2️⃣ Plot loss curves
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df_log["epoch"], df_log["train_loss"], label="Train", lw=2)
    plt.plot(df_log["epoch"], df_log["val_loss"], label="Validation", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (SmoothL1 Weighted)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if show: plt.show()

    # ---------------------------
    # 3️⃣ Evaluate model globally
    # ---------------------------
    if not (os.path.exists(model_path) and os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path)):
        print("⚠️ Model or scaler missing, skipping evaluation.")
        return

    X = torch.load(X_path)
    y = torch.load(y_path)

    # Load scalers
    x_scaler = torch.load(x_scaler_path)
    y_scaler = torch.load(y_scaler_path)

    # Normalize X as during training
    X = (X - x_scaler["mean"]) / x_scaler["std"]

    # reload model
    model_state = torch.load(model_path, map_location="cpu")["model_state"]
    model = make_model(X.shape[1])
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        preds_norm = model(X).squeeze().numpy()

    # denormalize predictions + targets
    y_true = y.squeeze().numpy()
    preds = preds_norm * y_scaler["std"].item() + y_scaler["mean"].item()

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    print(f"\n📈 Metrics on full dataset (real ΔCCI scale):")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")

    # ---------------------------
    # 4️⃣ Visuals
    # ---------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter true vs pred
    sns.scatterplot(x=y_true, y=preds, ax=axs[0], s=15, alpha=0.6)
    axs[0].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red", lw=2, linestyle="--"
    )
    axs[0].set_xlabel("True ΔCCI")
    axs[0].set_ylabel("Predicted ΔCCI")
    axs[0].set_title("True vs Predicted")

    # Histogram of residuals
    residuals = preds - y_true
    sns.histplot(residuals, bins=40, kde=True, ax=axs[1], color="steelblue")
    axs[1].axvline(0, color="red", linestyle="--")
    axs[1].set_title("Distribution of residuals")
    axs[1].set_xlabel("Prediction error (ΔCCI)")

    plt.tight_layout()
    if show: plt.show()

    # Distribution du ΔCCI réel
    y_all = torch.load(y_path).numpy()
    plt.hist(y_all, bins=50)
    plt.title("Distribution de ΔCCI")
    plt.show()

    # ---------------------------
    # 5️⃣ Optional: Save metrics
    # ---------------------------
    metrics_path = os.path.join(run_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"MAE  = {mae:.6f}\nRMSE = {rmse:.6f}\nR2    = {r2:.6f}\n")
    print(f"✅ Metrics saved to {metrics_path}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "residuals": residuals,
        "y_true": y_true,
        "y_pred": preds,
        "log": df_log,
    }


if __name__ == "__main__":
    # Exemple d'utilisation
    analyze_run("model/model_1")
