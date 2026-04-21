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
# File: check_progress.py
# Description: Monitor AlphaFold2 prediction progress and data completeness
# =============================================================================
#
# Overview
# --------
# Checks the completion status of AlphaFold2 predictions and identifies
# missing data. Used to track batch prediction progress.
#
# =============================================================================
# | Parameter   | Type  | Default | Description                          |
# |-------------|-------|---------|--------------------------------------|
# | --json      | flag  | False   | Output results in JSON format        |
# =============================================================================
#
# Input Requirements
# ------------------
# - Prediction outputs: af2_output/{name}/
# - Expected: 142 WT + 2050 Mut = 2192 total
#
# Output Format
# -------------
# - Console summary with progress percentage
# - Complete vs incomplete counts
# - Representation extraction status
#
# Usage Examples
# --------------
# 1. Check progress:
#    python check_progress.py
#
# 2. JSON output:
#    python check_progress.py --json
#
# =============================================================================

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import NamedTuple
OUTPUT_DIR = "/data/store-data/yeyh/scripts/AF2stability/af2_output"
DATA_DIR = "/data/store-data/yeyh/scripts/AF2stability"
TOTAL_WT = 142
TOTAL_MUT = 2050
TOTAL = TOTAL_WT + TOTAL_MUT


class ProgressStats(NamedTuple):
    """
    Progress statistics.
    """
    wt_found: int
    mut_found: int
    wt_complete: int
    mut_complete: int
    wt_repr: int
    mut_repr: int
    available_pairs: int
    progress_pct: float


def check_predictions() -> tuple[list[str], list[str]]:
    """
    Check which prediction directories exist.
    Returns
    -------
    tuple
        (wt_dirs, mut_dirs) lists
    """
    if not os.path.exists(OUTPUT_DIR):
        return [], []

    all_dirs = [
        d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]
    wt_dirs = [d for d in all_dirs if d.startswith("wt_")]
    mut_dirs = [d for d in all_dirs if d.startswith("mut_")]
    return wt_dirs, mut_dirs


def has_complete_output(name: str) -> bool:
    """
    Check if prediction has complete outputs.
    Parameters
    ----------
    name : str
        Directory name
    Returns
    -------
    bool
        True if both features.pkl and ranked_0.pdb exist
    """
    required_files = ["features.pkl", "ranked_0.pdb"]
    dir_path = os.path.join(OUTPUT_DIR, name)
    return all(os.path.exists(os.path.join(dir_path, f)) for f in required_files)


def check_completeness(wt_dirs: list, mut_dirs: list) -> tuple:
    """
    Check which predictions have complete outputs.
    Returns
    -------
    tuple
        (complete_wt, incomplete_wt, complete_mut, incomplete_mut)
    """
    complete_wt = [d for d in wt_dirs if has_complete_output(d)]
    incomplete_wt = [d for d in wt_dirs if not has_complete_output(d)]
    complete_mut = [d for d in mut_dirs if has_complete_output(d)]
    incomplete_mut = [d for d in mut_dirs if not has_complete_output(d)]
    return complete_wt, incomplete_wt, complete_mut, incomplete_mut


def check_representations(wt_dirs: list, mut_dirs: list) -> tuple[list, list]:
    """
    Check which have representations extracted.
    Returns
    -------
    tuple
        (wt_repr, mut_repr) lists
    """
    wt_repr = []
    for d in wt_dirs:
        repr_path = os.path.join(OUTPUT_DIR, d, "single_representation.npy")
        if os.path.exists(repr_path):
            wt_repr.append(d)
    mut_repr = []
    for d in mut_dirs:
        repr_path = os.path.join(OUTPUT_DIR, d, "single_representation.npy")
        if os.path.exists(repr_path):
            mut_repr.append(d)
    return wt_repr, mut_repr


def check_data_availability() -> dict:
    """
    Check which mutations have both WT and Mut predictions.
    Returns
    -------
    dict
        Mapping of mutation index to availability info
    """
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "fireprotdb_data_train.csv"))
        all_df = df
    except:
        return {}
    available = {}
    for idx in range(len(all_df)):
        mut_name = f"mut_{idx}"
        wt_name = f"wt_{idx // 100}"
        mut_path = os.path.join(OUTPUT_DIR, mut_name)
        if os.path.exists(mut_path) and has_complete_output(mut_name):
            available[idx] = {
                "mut": mut_name,
                "wt": wt_name,
                "position": all_df.iloc[idx]["position"] - 1,
                "ddg": all_df.iloc[idx]["ddg"],
            }
    return available


def get_stats() -> ProgressStats:
    """
    Calculate all progress statistics.
    Returns
    -------
    ProgressStats
        Named tuple with all statistics
    """
    wt_dirs, mut_dirs = check_predictions()
    complete_wt, _, complete_mut, _ = check_completeness(wt_dirs, mut_dirs)
    wt_repr, mut_repr = check_representations(wt_dirs, mut_dirs)
    available = check_data_availability()
    total_found = len(wt_dirs) + len(mut_dirs)
    progress_pct = total_found / TOTAL * 100
    return ProgressStats(
        wt_found=len(wt_dirs),
        mut_found=len(mut_dirs),
        wt_complete=len(complete_wt),
        mut_complete=len(complete_mut),
        wt_repr=len(wt_repr),
        mut_repr=len(mut_repr),
        available_pairs=len(available),
        progress_pct=progress_pct,
    )


def print_summary(stats: ProgressStats) -> None:
    print("AlphaFold2 Prediction Progress")
    print("-" * 60)

    print(f"\n{'Predictions Found':<25} {'Count':>10} {'Total':>10}")
    print("-" * 50)
    print(f"{'WT':<25} {stats.wt_found:>10} {TOTAL_WT:>10}")
    print(f"{'Mutant':<25} {stats.mut_found:>10} {TOTAL_MUT:>10}")
    print(f"{'Total':<25} {stats.wt_found + stats.mut_found:>10} {TOTAL:>10}")

    print(f"\n{'Complete Predictions (PDB)':<25} {'Count':>10}")
    print("-" * 40)
    print(f"{'WT':<25} {stats.wt_complete:>10}")
    print(f"{'Mutant':<25} {stats.mut_complete:>10}")

    print(f"\n{'Representations Extracted':<25} {'Count':>10}")
    print("-" * 40)
    print(f"{'WT':<25} {stats.wt_repr:>10}")
    print(f"{'Mutant':<25} {stats.mut_repr:>10}")

    print(f"\n{'Data Pairs Available':<25} {stats.available_pairs:>10}")

    print(f"\n{'Progress':<25} {stats.progress_pct:>10.1f}%")
    print(f"{'Remaining':<25} {TOTAL - stats.wt_found - stats.mut_found:>10}")


def print_json(stats: ProgressStats) -> None:
    """
    Print statistics as JSON.
    """
    result = {
        "predictions": {
            "wt_found": stats.wt_found,
            "mut_found": stats.mut_found,
            "total_found": stats.wt_found + stats.mut_found,
            "total_expected": TOTAL,
        },
        "complete": {
            "wt": stats.wt_complete,
            "mut": stats.mut_complete,
        },
        "representations": {
            "wt": stats.wt_repr,
            "mut": stats.mut_repr,
        },
        "data_pairs": stats.available_pairs,
        "progress_percent": stats.progress_pct,
        "remaining": TOTAL - stats.wt_found - stats.mut_found,
    }
    print(json.dumps(result, indent=2))




def main():
    parser = argparse.ArgumentParser(
        description="Check AlphaFold2 prediction progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s           # Show progress summary
  %(prog)s --json    # Output as JSON
        """,
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    stats = get_stats()
    if args.json:
        print_json(stats)
    else:
        print_summary(stats)





if __name__ == "__main__":
    main()
