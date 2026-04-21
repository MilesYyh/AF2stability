#!/usr/bin/env python
# '''
# @File    :   
# @Time    :   2026/04/06 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO    :   
# """
# =============================================================================
# File: train_model.py
# Description: MLP Training Pipeline for Protein Stability Prediction
# =============================================================================
#
# Overview
# --------
# Trains a multi-layer perceptron (MLP) to predict protein stability changes
# (ΔΔG) from AlphaFold2 single representations. Based on the paper:
# "Applications of AlphaFold beyond Protein Structure Prediction"
#
# Model Architecture
# ------------------
# Input: 1152-dimensional (384 × 3: WT + Mut + difference)
# Hidden: 1152 → 512 → 512 → Output: 1
# Activation: ReLU + Dropout (0.2)
#
# =============================================================================
# | Parameter      | Type   | Default  | Description                    |
# |----------------|--------|-----------|--------------------------------|
# | --epochs       | int    | 100       | Number of training epochs      |
# | --batch_size   | int    | 32        | Batch size                     |
# | --lr           | float  | 1e-4      | Learning rate                  |
# | --test         | flag   | False     | Test mode with mock data       |
# =============================================================================
#
# Input Requirements
# ------------------
# - Single representations: af2_output/{name}/single_representation.npy
# - Training data: fireprotdb_data_train.csv
#
# Output Format
# -------------
# - Model: models/mlp_model.pt
# - Metrics: models/mlp_model_metrics.pkl
#
# Usage Examples
# --------------
# 1. Train with default settings:
#    python train_model.py
#
# 2. Custom training parameters:
#    python train_model.py --epochs 200 --batch_size 64 --lr 1e-3
#
# 3. Test mode:
#    python train_model.py --test
#
# =============================================================================

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
OUTPUT_DIR = "/data/store-data/yeyh/scripts/AF2stability/af2_output"
DATA_DIR = "/data/store-data/yeyh/scripts/AF2stability"
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)



class StabilityDataset(Dataset):
    """
    PyTorch Dataset for stability prediction.
    """
    def __init__(self, features: np.ndarray, ddg: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.ddg = torch.FloatTensor(ddg)
    def __len__(self) -> int:
        return len(self.ddg)
    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.ddg[idx]


class MLPModel(nn.Module):
    """
    MLP model for ΔΔG prediction.
    Architecture (per paper):
    - Input: 1152 (384 × 3: WT + Mut + diff)
    - Hidden1: 1152 → 1152 + ReLU + Dropout(0.2)
    - Hidden2: 1152 → 512 + ReLU + Dropout(0.2)
    - Hidden3: 512 → 512 + ReLU + Dropout(0.2)
    - Output: 512 → 1
    """

    def __init__(self, input_dim: int = 1152):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1152),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)



def load_representations(name: str) -> np.ndarray | None:
    """
    Load single representation for a sequence.
    Parameters
    ----------
    name : str
        Directory name (e.g., "wt_0", "mut_100")
    Returns
    -------
    np.ndarray or None
        Single representation of shape (L, 384) if found
    """
    repr_path = os.path.join(OUTPUT_DIR, name, "single_representation.npy")
    if not os.path.exists(repr_path):
        return None
    return np.load(repr_path)


def get_mutation_features(
    wt_name: str, mut_name: str, position: int
) -> np.ndarray | None:
    """
    Build features for a mutation:
    - WT representation at mutation position
    - Mut representation at mutation position
    - Difference: Mut - WT
    Parameters
    ----------
    wt_name : str
        Wild-type name
    mut_name : str
        Mutant name
    position : int
        Mutation position (0-indexed)
    Returns
    -------
    np.ndarray or None
        1152-dimensional feature vector if successful
    """
    wt_repr = load_representations(wt_name)
    mut_repr = load_representations(mut_name)

    if wt_repr is None or mut_repr is None:
        return None
    L = min(wt_repr.shape[0], mut_repr.shape[0])
    if position >= L:
        return None
    # representations at mutation position
    wt_pos = wt_repr[position]  # (384,)
    mut_pos = mut_repr[position]  # (384,)
    diff = mut_pos - wt_pos  # (384,)
    # Concatenate: WT + Mut + diff = 1152
    features = np.concatenate([wt_pos, mut_pos, diff])
    return features


def load_all_data() -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Load all training data and build features.
    Returns
    -------
    tuple
        (features, ddg) where features is (N, 1152) and ddg is (N,)
    """

    train_df = pd.read_csv(os.path.join(DATA_DIR, "fireprotdb_data_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "fireprotdb_data_test.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "fireprotdb_data_validation.csv"))
    all_df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    # print(f"Total mutations: {len(all_df)}")
    unique_seqs = all_df["sequence"].unique()
    seq_to_wt_idx = {seq: idx for idx, seq in enumerate(unique_seqs)}

    # Build features for each mutation
    features_list = []
    ddg_list = []
    valid_indices = []
    error_count = 0
    for idx, row in all_df.iterrows():
        try:
            seq = row["sequence"]
            pos = row["position"] - 1  # Convert to 0-indexed
            wt_idx = seq_to_wt_idx.get(seq, -1)
            if wt_idx < 0:
                continue
            wt_name = f"wt_{wt_idx}"
            mut_name = f"mut_{idx}"
            # Build features
            features = get_mutation_features(wt_name, mut_name, pos)
            if features is not None:
                features_list.append(features)
                ddg_list.append(row["ddg"])
                valid_indices.append(idx)
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"Error at {idx}: {e}")

    if not features_list:
        print("No valid features found!")
        return None, None
    X = np.array(features_list, dtype=np.float32)
    y = np.array(ddg_list, dtype=np.float32)
    print(f"Built features: {X.shape}")
    print(f"DDG range: [{y.min():.2f}, {y.max():.2f}]")
    return X, y


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
) -> tuple[MLPModel, dict]:
    """
    Train MLP model with early stopping.
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (N, 1152)
    y : np.ndarray
        DDG labels (N,)
    test_size : float
        Fraction for validation set
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for DataLoader
    lr : float
        Learning rate
    Returns
    -------
    tuple
        (trained_model, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    # print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    train_dataset = StabilityDataset(X_train, y_train)
    test_dataset = StabilityDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = MLPModel(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # Training with early stopping
    best_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

    # best model
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_pred = model(X_test_tensor).numpy()
    pearson_r, pearson_p = pearsonr(y_test, y_pred)
    spearman_r, spearman_p = spearmanr(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"Test Results")
    print(f"{'=' * 50}")
    print(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")
    print(f"RMSE:       {rmse:.4f} kcal/mol")
    print(f"MAE:        {mae:.4f} kcal/mol")
    metrics = {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "pearson_p": pearson_p,
        "spearman_p": spearman_p,
        "rmse": rmse,
        "mae": mae,
    }
    return model, metrics


def save_model(model: MLPModel, metrics: dict, path: str) -> None:
    """
    Save model and metrics to disk.
    Parameters
    ----------
    model : MLPModel
        Trained PyTorch model
    metrics : dict
        Evaluation metrics
    path : str
        Output path for model weights
    """
    torch.save(model.state_dict(), path)
    with open(path.replace(".pt", "_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print(f"Model saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP model for protein stability prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Train with defaults
  %(prog)s --epochs 200 --batch_size 64    # Custom parameters
  %(prog)s --test                     # Test with mock data
        """,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode with simulated data"
    )
    args = parser.parse_args()
    if args.test:
        # Test mode: create mock data
        print("Running in TEST mode with mock data...")
        X = np.random.randn(100, 1152).astype(np.float32)
        y = np.random.randn(100).astype(np.float32) * 2.0
    else:
        # Load real data
        print("Loading data and building features...")
        X, y = load_all_data()

    if X is None:
        print("No data! Make sure predictions are complete.")
        return

    print(f"Training MLP")
    model, metrics = train_mlp(
        X, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    save_model(model, metrics, os.path.join(MODEL_DIR, "mlp_model.pt"))



if __name__ == "__main__":
    main()
