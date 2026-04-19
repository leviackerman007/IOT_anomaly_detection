from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


SQL_ROLLING_WINDOW = 5


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM autoencoder on the SWaT dataset.")
    parser.add_argument("--csv-path", default="SWaT_Dataset.csv", help="Path to the SWaT CSV file.")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length for the LSTM input.")
    parser.add_argument("--pca-components", type=int, default=20, help="Number of PCA components to keep.")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size for the LSTM encoder.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint-path", default="lstm_autoencoder_small.pth", help="Where to save weights.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for plots and metrics.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: str, artifacts_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    if "State" not in data.columns:
        raise ValueError("Expected a 'State' column in the dataset.")

    working_frame = data.copy()
    working_frame.insert(0, "row_id", np.arange(1, len(working_frame) + 1))

    feature_columns = [column for column in working_frame.columns if column not in {"row_id", "State", "t_stamp"}]
    rolling_source_columns = feature_columns[:5]

    connection = sqlite3.connect(":memory:")
    working_frame.to_sql("swat_raw", connection, index=False, if_exists="replace")

    raw_projection = ",\n            ".join(quote_identifier(column) for column in feature_columns)
    rolling_projection = ",\n            ".join(
        f'AVG({quote_identifier(column)}) OVER (ORDER BY row_id ROWS BETWEEN {SQL_ROLLING_WINDOW - 1} PRECEDING AND CURRENT ROW) AS {quote_identifier(f"{column}_ma{SQL_ROLLING_WINDOW}")}'
        for column in rolling_source_columns
    )

    engineered_query = f"""
        SELECT
            row_id,
            t_stamp,
            CASE TRIM(State) WHEN 'Attack' THEN 1 ELSE 0 END AS label,
            {raw_projection},
            {rolling_projection}
        FROM swat_raw
        ORDER BY row_id
    """

    connection.execute(f"CREATE TEMP VIEW engineered_swat AS {engineered_query}")

    summary_query = f"""
        SELECT
            label,
            COUNT(*) AS row_count,
            AVG({quote_identifier(rolling_source_columns[0])}) AS avg_{rolling_source_columns[0].lower()},
            AVG({quote_identifier(f'{rolling_source_columns[0]}_ma{SQL_ROLLING_WINDOW}')}) AS avg_{rolling_source_columns[0].lower()}_ma{SQL_ROLLING_WINDOW}
        FROM engineered_swat
        GROUP BY label
        ORDER BY label
    """

    feature_frame = pd.read_sql_query("SELECT * FROM engineered_swat", connection)
    summary_frame = pd.read_sql_query(summary_query, connection)
    summary_frame.to_csv(artifacts_dir / "sql_feature_profile.csv", index=False)

    labels = feature_frame["label"].to_numpy()
    feature_frame = feature_frame.drop(columns=["row_id", "t_stamp", "label"])

    return feature_frame.to_numpy(), labels, feature_frame


def create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    sequences = [data[index : index + seq_len] for index in range(len(data) - seq_len)]
    return np.asarray(sequences)


def prepare_data(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, PCA]:
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(scaled_features)

    sequences = create_sequences(reduced_features, seq_len)
    sequence_labels = labels[seq_len:]

    return sequences, sequence_labels, reduced_features, scaler, pca


def build_loader(sequences: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / max(1, len(loader))
        loss_history.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.6f}")

    return loss_history


@torch.no_grad()
def reconstruction_errors(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_errors = []

    for (batch,) in loader:
        batch = batch.to(device)
        reconstruction = model(batch)
        mse = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
        all_errors.append(mse.cpu().numpy())

    return np.concatenate(all_errors)


def select_threshold(labels: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(errors.min(), errors.max(), 100)
    best_threshold = thresholds[0]
    best_f1 = 0.0

    for threshold in thresholds:
        predictions = (errors > threshold).astype(int)
        current_f1 = f1_score(labels, predictions)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_reconstruction_errors(errors: np.ndarray, labels: np.ndarray, threshold: float, artifacts_dir: Path) -> None:
    predictions = (errors > threshold).astype(int)
    anomaly_indices = np.where(predictions == 1)[0]

    plt.figure(figsize=(12, 5))
    plt.plot(errors, label="Reconstruction Error", color="blue")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.6f}")
    plt.scatter(anomaly_indices, errors[anomaly_indices], color="orange", s=10, label="Detected Anomalies")
    plt.title("LSTM Autoencoder Reconstruction Error")
    plt.xlabel("Sequence Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(artifacts_dir / "reconstruction_error.png")

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color="skyblue")
    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Reconstruction Errors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(artifacts_dir / "error_histogram.png")

    precision_values, recall_values, _ = precision_recall_curve(labels, errors)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_values, precision_values)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    save_plot(artifacts_dir / "precision_recall_curve.png")

    fpr, tpr, _ = roc_curve(labels, errors)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}", color="red")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(artifacts_dir / "roc_curve.png")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    features, labels, _ = load_dataset(args.csv_path, artifacts_dir=artifacts_dir)
    sequences, sequence_labels, _, _, _ = prepare_data(
        features=features,
        labels=labels,
        seq_len=args.seq_len,
        pca_components=args.pca_components,
    )

    train_sequences = sequences[sequence_labels == 0]
    train_loader = build_loader(train_sequences, batch_size=args.batch_size, shuffle=True)
    eval_loader = build_loader(sequences, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(n_features=sequences.shape[2], hidden_size=args.hidden_size).to(device)

    loss_history = train_model(model, train_loader, device=device, epochs=args.epochs, lr=args.lr)
    torch.save(model.state_dict(), args.checkpoint_path)

    errors = reconstruction_errors(model, eval_loader, device=device)
    aligned_labels = sequence_labels[: len(errors)]

    valid_mask = ~np.isnan(aligned_labels)
    aligned_labels = aligned_labels[valid_mask]
    errors = errors[valid_mask]

    best_threshold, best_f1 = select_threshold(aligned_labels, errors)
    predictions = (errors > best_threshold).astype(int)

    precision = precision_score(aligned_labels, predictions)
    recall = recall_score(aligned_labels, predictions)
    roc_auc = roc_auc_score(aligned_labels, errors)

    metrics = {
        "threshold": float(best_threshold),
        "best_f1": float(best_f1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(aligned_labels, predictions)),
        "roc_auc": float(roc_auc),
        "final_loss": float(loss_history[-1]) if loss_history else None,
        "epochs": args.epochs,
        "seq_len": args.seq_len,
        "pca_components": args.pca_components,
    }

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plot_reconstruction_errors(errors, aligned_labels, best_threshold, artifacts_dir)

    print("\nFinal metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()