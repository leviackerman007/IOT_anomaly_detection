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

