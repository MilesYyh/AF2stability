#!/usr/bin/env python
# '''
# @File    :
# @Time    :   2026/08/17 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO
# """

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, cast
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from pandas import DataFrame, Series
VERSION = '1.0.0'
ONE_KCAL_MOL_IN_KJ = 4.184


def resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    return repo_root / path_str


def load_predictions(predictions_csv: Path, split: str) -> DataFrame:
    frame = pd.read_csv(predictions_csv)
    return frame.loc[frame['split'] == split].copy()


def load_summary(summary_json: Path) -> dict[str, Any]:
    return json.loads(summary_json.read_text(encoding='utf-8'))


def add_identity_line(ax: Axes, values_x: Series, values_y: Series) -> None:
    lower = min(values_x.min(), values_y.min())
    upper = max(values_x.max(), values_y.max())
    ax.plot([lower, upper], [lower, upper], linestyle='--', linewidth=1.2, color='black', alpha=0.7)


def add_regression_line(ax: Axes, values_x: Series, values_y: Series) -> None:
    slope, intercept = np.polyfit(values_x, values_y, 1)
    lower = min(values_x.min(), values_y.min())
    upper = max(values_x.max(), values_y.max())
    x_line = np.array([lower, upper], dtype=float)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='tab:red', alpha=0.8, linewidth=1.8)


def add_gray_band(ax: Axes, values_x: Series, values_y: Series) -> None:
    lower = min(values_x.min(), values_y.min())
    upper = max(values_x.max(), values_y.max())
    x_line = np.array([lower, upper], dtype=float)
    ax.fill_between(
        x_line,
        x_line - ONE_KCAL_MOL_IN_KJ,
        x_line + ONE_KCAL_MOL_IN_KJ,
        color='gray',
        alpha=0.12,
    )


def build_stat_text(subtitle: str) -> str:
    return subtitle.replace(', ', '\n')


def make_scatter_plot(
    *,
    frame: DataFrame,
    title: str,
    subtitle: str,
    output_path: Path,
    color_column: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=300)
    x_values = cast(Series, frame['y_true'])
    y_values = cast(Series, frame['y_pred'])

    if color_column in frame.columns:
        grouped = frame.groupby(color_column)
        for group_name, group_df in grouped:
            ax.scatter(
                group_df['y_true'],
                group_df['y_pred'],
                s=26,
                alpha=0.6,
                marker='s',
                edgecolors='none',
                label=str(group_name),
            )
        if frame[color_column].nunique() > 1:
            ax.legend(title=color_column, fontsize=8, title_fontsize=9, frameon=False)
    else:
        ax.scatter(frame['y_true'], frame['y_pred'], s=26, alpha=0.6, marker='s', edgecolors='none')

    add_gray_band(ax, x_values, y_values)
    add_identity_line(ax, x_values, y_values)
    add_regression_line(ax, x_values, y_values)
    ax.set_xlabel('Exp ΔΔG', fontsize=11)
    ax.set_ylabel('Pred ΔΔG', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.25, linestyle=':')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(
        xmin + (xmax - xmin) * 0.05,
        ymax - (ymax - ymin) * 0.05,
        build_stat_text(subtitle),
        verticalalignment='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot paper-like predicted-vs-experimental ddG scatter figures.'
    )
    parser.add_argument(
        '--holdout-predictions',
        default='results/models/ddg_mlp_full_partial_holdout_predictions.csv',
        help='Holdout prediction CSV path',
    )
    parser.add_argument(
        '--holdout-summary',
        default='results/models/ddg_mlp_full_partial_holdout_summary.json',
        help='Holdout summary JSON path',
    )
    parser.add_argument(
        '--cv-predictions',
        default='results/models/ddg_mlp_full_partial_cv10_predictions.csv',
        help='CV prediction CSV path',
    )
    parser.add_argument(
        '--cv-summary',
        default='results/models/ddg_mlp_full_partial_cv10_summary.json',
        help='CV summary JSON path',
    )
    parser.add_argument(
        '--output-dir',
        default='results/figures/ddg',
        help='Output directory for scatter figures',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    holdout_predictions = resolve_repo_path(repo_root, args.holdout_predictions)
    holdout_summary = resolve_repo_path(repo_root, args.holdout_summary)
    cv_predictions = resolve_repo_path(repo_root, args.cv_predictions)
    cv_summary = resolve_repo_path(repo_root, args.cv_summary)
    output_dir = resolve_repo_path(repo_root, args.output_dir)

    holdout_frame = load_predictions(holdout_predictions, split='val')
    holdout_stats = load_summary(holdout_summary)
    holdout_subtitle = (
        f"r = {float(holdout_stats['val_metrics']['pearson']):.3f}, "
        f"MAE = {float(holdout_stats['val_metrics']['mae']):.3f}, "
        f"RMSE = {float(holdout_stats['val_metrics']['rmse']):.3f}"
    )
    make_scatter_plot(
        frame=holdout_frame,
        title='Holdout predicted vs experimental ΔΔG',
        subtitle=holdout_subtitle,
        output_path=output_dir / 'holdout_predicted_vs_experimental_ddg.png',
        color_column='split',
    )

    cv_frame = load_predictions(cv_predictions, split='val')
    cv_stats = load_summary(cv_summary)
    cv_subtitle = (
        f"r = {float(cv_stats['aggregate_metrics']['val_pearson_mean']):.3f}, "
        f"MAE = {float(cv_stats['aggregate_metrics']['val_mae_mean']):.3f}, "
        f"RMSE = {float(cv_stats['aggregate_metrics']['val_rmse_mean']):.3f}"
    )
    make_scatter_plot(
        frame=cv_frame,
        title='Random 10-fold CV predicted vs experimental ΔΔG',
        subtitle=cv_subtitle,
        output_path=output_dir / 'cv10_predicted_vs_experimental_ddg.png',
        color_column='fold_index',
    )

    summary = {
        'version': VERSION,
        'holdout_points': int(len(holdout_frame)),
        'cv_val_points': int(len(cv_frame)),
        'output_dir': str(output_dir),
        'files': [
            str(output_dir / 'holdout_predicted_vs_experimental_ddg.png'),
            str(output_dir / 'cv10_predicted_vs_experimental_ddg.png'),
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
