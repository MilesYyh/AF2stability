#!/usr/bin/env python
# '''
# @File    :   
# @Time    :   2026/04/21 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO    :   
# """
# =============================================================================
# File: visualize_results.py
# Description: Visualization tools for stability prediction results
# =============================================================================
#
# Overview
# --------
# Generates plots and reports for evaluating protein stability prediction
# model performance. Includes prediction vs truth, DDG distribution, and
# residual analysis.
#
# =============================================================================
# | Parameter    | Type  | Default | Description                      |
# |--------------|-------|---------|----------------------------------|
# | --report     | flag  | False   | Generate full analysis report    |
# | --sample     | flag  | False   | Generate demo plots with synthetic data |
# =============================================================================
#
# Input Requirements
# ------------------
# - Predictions: results/predictions.npy
# - Labels: results/labels.npy
#
# Output Format
# -------------
# - prediction_vs_truth.png: Scatter plot of predicted vs experimental DDG
# - ddg_distribution.png: Histogram of DDG values
# - residuals.png: Residual analysis plots
# - training_history.png: Loss curves during training
#
# Usage Examples
# --------------
# 1. Generate demo plots:
#    python visualize_results.py --sample
#
# 2. Generate full report:
#    python visualize_results.py --report
#
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
DATA_DIR = "/data/store-data/yeyh/scripts/AF2stability"
OUTPUT_DIR = os.path.join(DATA_DIR, "af2_output")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)




def load_predictions_and_labels() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load predicted DDG and ground truth labels."""
    pred_path = os.path.join(RESULTS_DIR, "predictions.npy")
    label_path = os.path.join(RESULTS_DIR, "labels.npy")
    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        return None, None
    return np.load(pred_path), np.load(label_path)


def plot_prediction_vs_truth(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Prediction vs Ground Truth"
) -> None:
    """Plot predicted DDG vs experimental DDG."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ax.set_xlabel("Experimental ΔΔG (kcal/mol)")
    ax.set_ylabel("Predicted ΔΔG (kcal/mol)")
    ax.set_title(
        f"{title}\nPearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}, RMSE = {rmse:.3f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "prediction_vs_truth.png"), dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/prediction_vs_truth.png")


def plot_ddg_distribution(y_true: np.ndarray, y_pred: np.ndarray | None = None) -> None:
    """Plot DDG distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_true, bins=50, alpha=0.7, label="Experimental", color="blue")
    if y_pred is not None:
        ax.hist(y_pred, bins=50, alpha=0.5, label="Predicted", color="red")
    ax.set_xlabel("ΔΔG (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("DDG Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ddg_distribution.png"), dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/ddg_distribution.png")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot prediction residuals."""
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(residuals, bins=50, alpha=0.7, color="green")
    axes[0].axvline(x=0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (Predicted - Experimental)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual Distribution")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted ΔΔG")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predictions")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals.png"), dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/residuals.png")


def plot_training_history(history: dict) -> None:
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/training_history.png")


def create_summary_report() -> None:
    """Generate full analysis report with all visualizations."""
    y_true, y_pred = load_predictions_and_labels()
    if y_true is None:
        print("No predictions yet - skipping visualizations")
        print("Run train_model.py first to generate predictions")
        return
    plot_prediction_vs_truth(y_true, y_pred)
    plot_ddg_distribution(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)
    print(f"Pearson r:  {pearson_r:.4f}")
    print(f"Spearman ρ: {spearman_r:.4f}")
    print(f"RMSE:       {rmse:.4f} kcal/mol")
    print(f"MAE:        {mae:.4f} kcal/mol")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize stability prediction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sample    # Generate demo plots
  %(prog)s --report    # Generate full report
        """,
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate full analysis report"
    )
    parser.add_argument(
        "--sample", action="store_true", help="Generate demo plots with synthetic data"
    )
    args = parser.parse_args()
    if args.report:
        create_summary_report()
    elif args.sample:
        print("Generating sample plots...")
        np.random.seed(42)
        y_true = np.random.randn(1000)
        y_pred = y_true + np.random.randn(1000) * 0.5
        plot_prediction_vs_truth(y_true, y_pred, "Sample: Test Set")
        plot_ddg_distribution(y_true, y_pred)
        plot_residuals(y_true, y_pred)
        print("Sample plots saved!")
    else:
        print("Use --report for results or --sample for demo plots")





if __name__ == "__main__":
    main()
