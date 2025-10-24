# 🇺🇸 ΔCCI Forecasting Pipeline (US)

This project builds a deep learning pipeline for predicting monthly **ΔCCI (Consumer Confidence Index change)** from **GDELT news features** using a sequence-to-one LSTM with attention.

---

## Environment Setup

The project uses [`uv`](https://github.com/astral-sh/uv) (a fast Python package manager).
Make sure you have it installed first:


Install and sync all dependencies:

```bash
uv sync
```

This will automatically create a `.venv` and install all required packages defined in `pyproject.toml`.

---

## Pipeline Overview

The end-to-end process consists of the following steps:

### Fetch GDELT Data

Download and store all relevant GDELT event and tone features.

```bash
uv run src/gdelt_fetch.py
```

This script pulls the data and saves the processed `.parquet` files into the `data/` directory.

---

### Preprocess the US Dataset

Prepare the time-series tensors used for model training.

```bash
uv run src/preprocess_us.py
```

This generates:

* `data/X_us.pt` → model input sequences
* `data/y_us.pt` → ΔCCI targets

Both tensors are used by the training script.

---

### Train the Model

Train the ΔCCI prediction model (many-to-one, 60-day input window).

```bash
uv run src/train.py
```

The model, logs, scalers, and plots are automatically saved under:

```
models/model_<id>/
```

---

### Optional: Analyze a Trained Run

You can visualize or evaluate any saved training run using:

```bash
uv run analyze_run_us.py
```

> **Note:** Before running, open the file and update the path to the desired model directory (e.g. `models/model_4/`).

---

## Directory Structure

```
├── src/
│   ├── gdelt_fetch.py
│   ├── preprocess_us.py
│   ├── train.py
│   └── analyze_run_us.py
├── data/
│   ├── X_us.pt
│   ├── y_us.pt
│   └── ...
├── models/
│   ├── model_0/
│   ├── model_1/
│   └── ...
└── pyproject.toml
```

---

## Notes

* Training and validation are split chronologically to prevent data leakage.
* Normalization parameters (`scalers.pt`) are saved and must be reused for inference.
* GPU (CUDA) is automatically used if available.

---
