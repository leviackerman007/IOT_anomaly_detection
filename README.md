<<<<<<< HEAD
# Distributed IoT Anomaly Detection

This project implements a **distributed LSTM Autoencoder pipeline** for unsupervised anomaly detection on multivariate IoT time series data. It combines deep learning with scalable workflows and SQL-based feature engineering to evaluate sensor health across large streams.

## ✨ Features
- **Sliding-window segmentation** for temporal context
- **PCA preprocessing** to reduce dimensionality
- **LSTM Autoencoder** for sequence reconstruction
- **Reconstruction-error thresholding** for anomaly detection
- **Batch inference workflows** for distributed evaluation
- **SQL integration** for:
  - Windowed joins
  - Lag functions
  - Automated sensor-health reporting
  - Seamless relational queries with deep learning inference

## 📊 Results
- Achieved **99.57% precision** on benchmark IoT datasets
- Scalable inference across distributed sensor streams

=======
# SWaT LSTM Autoencoder for Industrial Anomaly Detection

Anomaly detection pipeline for the **Secure Water Treatment (SWaT)** dataset using a **PCA-compressed LSTM autoencoder**. The project learns normal system behavior from multivariate sensor and actuator readings, then flags suspicious activity through reconstruction error thresholding.

## Why This Project Matters

Industrial control systems generate high-dimensional time-series data that is difficult to monitor manually. This project is relevant for real-world monitoring because it combines:

- sequence modeling for time-dependent behavior,
- dimensionality reduction to keep training practical,
- unsupervised anomaly detection when labeled attacks are limited,
- threshold-based decisioning that is easy to deploy and explain.

That makes it a good resume project for roles involving machine learning, time-series analytics, anomaly detection, or industrial data systems.

## What It Does

1. Loads the SWaT dataset.
2. Maps `State` to binary labels: `Normal = 0`, `Attack = 1`.
3. Scales sensor features with `MinMaxScaler`.
4. Uses SQLite and SQL window functions to stage the dataset and engineer rolling features.
5. Reduces feature dimensionality with PCA.
6. Builds fixed-length sequences for LSTM input.
7. Trains an LSTM autoencoder on the sequence data.
8. Uses reconstruction error to detect anomalies.
9. Chooses a threshold by maximizing F1 score.
10. Reports precision, recall, F1, ROC-AUC, and visualization plots.

## Model Overview

- **Input**: Multivariate time-series windows from SWaT.
- **Preprocessing**: MinMax scaling + PCA compression.
- **SQL layer**: SQLite staging, label extraction, and rolling mean feature engineering.
- **Sequence length**: 30 timesteps.
- **Model**: LSTM autoencoder.
- **Loss**: Mean squared error reconstruction loss.
- **Decision rule**: `reconstruction_error > threshold`.

The current notebook snapshot shows an optimal threshold around **0.0294** in the saved plots.

## Repository Contents

- `train_swat_autoencoder.py` - reusable training script with saved metrics and plots.
- `model.ipynb` - notebook prototype with preprocessing, training, evaluation, and plots.
- `SWaT_Dataset.csv` - local dataset file expected at runtime, not tracked in git.
- `lstm_autoencoder*.pth` - local saved model checkpoints, not tracked in git.
- `artifacts/` - generated plots and `metrics.json` from the training script.
- `artifacts/sql_feature_profile.csv` - SQL-generated label and feature profile.
- `requirements.txt` - Python dependencies.

## Getting Started

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter lab model.ipynb
```

Run the notebook cells from top to bottom to reproduce preprocessing, training, and anomaly scoring.

Before running, place `SWaT_Dataset.csv` in the project root. The dataset and trained weights are intentionally excluded from version control so the repository stays lightweight and pushable.

### 4. Run the script version

```bash
python train_swat_autoencoder.py
```

This creates a checkpoint plus reproducible outputs in `artifacts/`:

- `metrics.json`
- `sql_feature_profile.csv`
- `reconstruction_error.png`
- `error_histogram.png`
- `precision_recall_curve.png`
- `roc_curve.png`

## Data Notes

- The notebook expects `SWaT_Dataset.csv` in the project root.
- The code drops the timestamp column and uses the `State` column for evaluation labels.
- If your dataset schema differs, update the column names in the preprocessing cell before running.

## Results

The notebook and script currently produce:

- SQL-based feature profiling and rolling-window feature engineering,
- reconstruction error plots,
- precision/recall vs threshold curves,
- ROC curve,
- anomaly markers on the error timeline,
- saved model weights in `.pth` format.

If you want this to look stronger on a resume, the next step is to paste the final numbers from `artifacts/metrics.json` into a small results table.

## Suggested Resume Framing

Use a description like this:

> Built an unsupervised anomaly detection pipeline for industrial control system telemetry using SQLite/SQL feature engineering, PCA-compressed LSTM autoencoder modeling, and reconstruction-error thresholding on the SWaT dataset. Implemented sequence modeling and evaluation with precision, recall, F1, and ROC-AUC, with reproducible metrics and plots saved to disk.

## Future Improvements

- Move the notebook logic into a reusable training script.
- Add a small CLI for training and inference.
- Log experiments and metrics to a JSON or CSV report.
- Add tests for preprocessing and threshold selection.
- Compare against alternative detectors such as isolation forest or one-class SVM.
>>>>>>> fb5e99c (Updating filrs)
