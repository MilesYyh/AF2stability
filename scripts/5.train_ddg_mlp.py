#!/usr/bin/env python
# '''
# @File    :
# @Time    :   2026/07/02 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO
# """

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
VERSION = '1.0.0'


class DdGMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1152, 1152),
            nn.ReLU(),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    y_true_std = float(np.std(y_true))
    y_pred_std = float(np.std(y_pred))
    if y_true.shape[0] < 2 or y_true_std == 0.0 or y_pred_std == 0.0:
        pearson = None
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson,
    }

def require_metric(metrics: dict[str, float | None], key: str) -> float:
    value = metrics[key]
    if value is None:
        raise ValueError(f'Metric {key} is unexpectedly None')
    return float(value)

def train_epoch(
    model: DdGMlp,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
    return epoch_loss / len(loader)

def predict(
    model: DdGMlp,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(-1))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    preds: list[np.ndarray] = []
    loss_fn = nn.SmoothL1Loss()
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            total_loss += float(loss.item())
            preds.append(pred.cpu().numpy().reshape(-1))

    predictions = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
    return predictions, total_loss / len(loader)


def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(-1))
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle)

def shuffled_indices(num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(num_samples)

def split_train_val_indices(num_samples: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f'val_fraction must be between 0 and 1, got {val_fraction}')
    indices = shuffled_indices(num_samples, seed)
    val_size = max(1, int(round(num_samples * val_fraction)))
    if val_size >= num_samples:
        val_size = num_samples - 1
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]
    return train_idx, val_idx

def build_kfold_indices(num_samples: int, num_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if num_folds < 2:
        raise ValueError(f'num_folds must be >= 2, got {num_folds}')
    if num_folds > num_samples:
        raise ValueError(f'num_folds ({num_folds}) cannot exceed num_samples ({num_samples})')

    indices = shuffled_indices(num_samples, seed)
    fold_sizes = np.full(num_folds, num_samples // num_folds, dtype=int)
    fold_sizes[: num_samples % num_folds] += 1

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current: current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]], axis=0)
        folds.append((train_idx, val_idx))
        current += fold_size
    return folds

def resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    return repo_root / path_str

def load_feature_dataset(input_npz: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(input_npz, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    sample_ids = data['sample_ids'].tolist()
    return X, y, sample_ids

def write_summary(summary_json: Path, summary: dict[str, object]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

def write_prediction_rows(predictions_csv: Path, rows: list[dict[str, object]]) -> None:
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'mode',
        'split',
        'fold_index',
        'sample_id',
        'y_true',
        'y_pred',
    ]
    with predictions_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train or minimally validate the ddG MLP on extracted single-representation features.'
    )
    parser.add_argument(
        '--input-npz',
        default='data/processed/single_repr_features_smoketest.npz',
        help='Input NPZ containing X, y, sample_ids',
    )
    parser.add_argument(
        '--summary-json',
        default='results/models/ddg_mlp_smoketest_summary.json',
        help='Output summary JSON path',
    )
    parser.add_argument(
        '--state-dict',
        default='results/models/ddg_mlp_smoketest_state_dict.pt',
        help='Output model state dict path',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for training/validation',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Adam learning rate',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Target epoch count for full training mode',
    )
    parser.add_argument(
        '--mode',
        choices=['train_only', 'holdout', 'cv'],
        default='train_only',
        help='Run train-only fitting, single holdout evaluation, or K-fold CV',
    )
    parser.add_argument(
        '--val-fraction',
        type=float,
        default=0.2,
        help='Validation fraction for holdout mode',
    )
    parser.add_argument(
        '--num-folds',
        type=int,
        default=5,
        help='Number of folds for CV mode',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for dataset splitting',
    )
    parser.add_argument(
        '--include-sample-ids',
        action='store_true',
        help='Include sample ID lists in the summary JSON',
    )
    parser.add_argument(
        '--predictions-csv',
        default='',
        help='Optional CSV path for per-sample prediction outputs',
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    input_npz = resolve_repo_path(repo_root, args.input_npz)
    summary_json = resolve_repo_path(repo_root, args.summary_json)
    state_dict_path = resolve_repo_path(repo_root, args.state_dict)
    predictions_csv = resolve_repo_path(repo_root, args.predictions_csv) if args.predictions_csv else None

    X, y, sample_ids = load_feature_dataset(input_npz)
    prediction_rows: list[dict[str, object]] = []

    if X.ndim != 2 or X.shape[1] != 1152:
        raise ValueError(f'Expected feature matrix shape (N, 1152), got {X.shape}')
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f'Label shape mismatch: X={X.shape}, y={y.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DdGMlp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(-1))
    loader = build_loader(X, y, args.batch_size, shuffle=True)

    summary: dict[str, object] = {
        'version': VERSION,
        'input_npz': str(input_npz),
        'num_samples': int(X.shape[0]),
        'feature_dim': int(X.shape[1]),
        'device': str(device),
        'learning_rate': args.learning_rate,
        'batch_size_requested': args.batch_size,
        'epochs_requested': args.epochs,
        'run_mode_requested': args.mode,
        'seed': args.seed,
        'include_sample_ids': args.include_sample_ids,
    }
    if args.include_sample_ids:
        summary['sample_ids'] = sample_ids

    if len(dataset) < 2:
        model.train()
        batch_x, batch_y = next(iter(loader))
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        pred_before = model(batch_x)
        loss_before = loss_fn(pred_before, batch_y)
        loss_before.backward()
        optimizer.step()
        with torch.no_grad():
            pred_after = model(batch_x)
            loss_after = loss_fn(pred_after, batch_y)

        summary.update(
            {
                'mode': 'smoketest_minimal_train_step',
                'loss_before_step': float(loss_before.item()),
                'loss_after_step': float(loss_after.item()),
                'prediction_before_step': pred_before.detach().cpu().numpy().reshape(-1).tolist(),
                'prediction_after_step': pred_after.detach().cpu().numpy().reshape(-1).tolist(),
                'message': 'Dataset too small for meaningful training; validated forward/backward/update path instead.',
            }
        )
    elif args.mode == 'train_only':
        losses = []
        for _ in range(args.epochs):
            losses.append(train_epoch(model, loader, optimizer, loss_fn, device))

        summary.update(
            {
                'mode': 'train_only',
                'final_loss': losses[-1],
                'min_loss': min(losses),
            }
        )
    elif args.mode == 'holdout':
        train_idx, val_idx = split_train_val_indices(len(dataset), args.val_fraction, args.seed)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        train_loader = build_loader(X_train, y_train, args.batch_size, shuffle=True)
        train_losses = []
        for _ in range(args.epochs):
            train_losses.append(train_epoch(model, train_loader, optimizer, loss_fn, device))

        train_pred, train_eval_loss = predict(model, X_train, y_train, args.batch_size, device)
        val_pred, val_eval_loss = predict(model, X_val, y_val, args.batch_size, device)
        summary.update(
            {
                'mode': 'holdout',
                'train_size': int(X_train.shape[0]),
                'val_size': int(X_val.shape[0]),
                'val_fraction': args.val_fraction,
                'final_train_loss': train_losses[-1],
                'min_train_loss': min(train_losses),
                'train_eval_loss': train_eval_loss,
                'val_eval_loss': val_eval_loss,
                'train_metrics': compute_regression_metrics(y_train, train_pred),
                'val_metrics': compute_regression_metrics(y_val, val_pred),
            }
        )
        if predictions_csv is not None:
            for idx, sample_idx in enumerate(train_idx.tolist()):
                prediction_rows.append(
                    {
                        'mode': 'holdout',
                        'split': 'train',
                        'fold_index': 1,
                        'sample_id': sample_ids[sample_idx],
                        'y_true': float(y_train[idx]),
                        'y_pred': float(train_pred[idx]),
                    }
                )
            for idx, sample_idx in enumerate(val_idx.tolist()):
                prediction_rows.append(
                    {
                        'mode': 'holdout',
                        'split': 'val',
                        'fold_index': 1,
                        'sample_id': sample_ids[sample_idx],
                        'y_true': float(y_val[idx]),
                        'y_pred': float(val_pred[idx]),
                    }
                )
        if args.include_sample_ids:
            summary['train_sample_ids'] = [sample_ids[i] for i in train_idx.tolist()]
            summary['val_sample_ids'] = [sample_ids[i] for i in val_idx.tolist()]
    else:
        folds = build_kfold_indices(len(dataset), args.num_folds, args.seed)
        fold_summaries = []
        cv_metrics: dict[str, list[float]] = {
            'train_eval_loss': [],
            'val_eval_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
        }
        cv_pearson: dict[str, list[float]] = {
            'train_pearson': [],
            'val_pearson': [],
        }

        for fold_index, (train_idx, val_idx) in enumerate(folds, start=1):
            fold_model = DdGMlp().to(device)
            fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=args.learning_rate)
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            train_loader = build_loader(X_train, y_train, args.batch_size, shuffle=True)

            train_losses = []
            for _ in range(args.epochs):
                train_losses.append(train_epoch(fold_model, train_loader, fold_optimizer, loss_fn, device))

            train_pred, train_eval_loss = predict(fold_model, X_train, y_train, args.batch_size, device)
            val_pred, val_eval_loss = predict(fold_model, X_val, y_val, args.batch_size, device)
            train_metrics = compute_regression_metrics(y_train, train_pred)
            val_metrics = compute_regression_metrics(y_val, val_pred)

            cv_metrics['train_eval_loss'].append(train_eval_loss)
            cv_metrics['val_eval_loss'].append(val_eval_loss)
            cv_metrics['train_mae'].append(require_metric(train_metrics, 'mae'))
            cv_metrics['val_mae'].append(require_metric(val_metrics, 'mae'))
            cv_metrics['train_rmse'].append(require_metric(train_metrics, 'rmse'))
            cv_metrics['val_rmse'].append(require_metric(val_metrics, 'rmse'))
            if train_metrics['pearson'] is not None:
                cv_pearson['train_pearson'].append(float(train_metrics['pearson']))
            if val_metrics['pearson'] is not None:
                cv_pearson['val_pearson'].append(float(val_metrics['pearson']))

            fold_summaries.append(
                {
                    'fold_index': fold_index,
                    'train_size': int(X_train.shape[0]),
                    'val_size': int(X_val.shape[0]),
                    'final_train_loss': train_losses[-1],
                    'min_train_loss': min(train_losses),
                    'train_eval_loss': train_eval_loss,
                    'val_eval_loss': val_eval_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }
            )

            if predictions_csv is not None:
                for idx, sample_idx in enumerate(train_idx.tolist()):
                    prediction_rows.append(
                        {
                            'mode': 'cv',
                            'split': 'train',
                            'fold_index': fold_index,
                            'sample_id': sample_ids[sample_idx],
                            'y_true': float(y_train[idx]),
                            'y_pred': float(train_pred[idx]),
                        }
                    )
                for idx, sample_idx in enumerate(val_idx.tolist()):
                    prediction_rows.append(
                        {
                            'mode': 'cv',
                            'split': 'val',
                            'fold_index': fold_index,
                            'sample_id': sample_ids[sample_idx],
                            'y_true': float(y_val[idx]),
                            'y_pred': float(val_pred[idx]),
                        }
                    )

        aggregate_metrics: dict[str, float | None] = {}
        for name, values in cv_metrics.items():
            aggregate_metrics[f'{name}_mean'] = float(np.mean(values))
            aggregate_metrics[f'{name}_std'] = float(np.std(values))
        for name, values in cv_pearson.items():
            if values:
                aggregate_metrics[f'{name}_mean'] = float(np.mean(values))
                aggregate_metrics[f'{name}_std'] = float(np.std(values))
            else:
                aggregate_metrics[f'{name}_mean'] = None
                aggregate_metrics[f'{name}_std'] = None

        summary.update(
            {
                'mode': 'cv',
                'num_folds': args.num_folds,
                'folds': fold_summaries,
                'aggregate_metrics': aggregate_metrics,
            }
        )

    state_dict_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_dict_path)
    summary['state_dict_path'] = str(state_dict_path)

    if predictions_csv is not None:
        write_prediction_rows(predictions_csv, prediction_rows)
        summary['predictions_csv'] = str(predictions_csv)

    write_summary(summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))



if __name__ == '__main__':
    main()
