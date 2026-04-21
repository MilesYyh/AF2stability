#!/usr/bin/env python
# '''
# @File    :   
# @Time    :   2026/04/07 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO    :   
# """
# =============================================================================
# File: sequence_alignment.py
# Description: Handle sequence length alignment for AlphaFold2 representations
# =============================================================================
#
# Overview
# --------
# Different proteins have different sequence lengths. This script handles
# the alignment and feature extraction for training the stability prediction
# model.
#
# Key Insight
# -----------
# For mutation prediction, we only need the representation at the mutation
# position, not the entire sequence. This avoids padding/truncation issues.
#
# =============================================================================
# | Parameter      | Type   | Default | Description                      |
# |----------------|--------|---------|----------------------------------|
# | --analyze      | flag   | True    | Analyze sequence length dist     |
# | --build        | flag   | False   | Build training dataset           |
# =============================================================================
#
# Input Requirements
# ------------------
# - Representations: af2_output/{name}/single_representation.npy
# - Mutation data: fireprotdb_data_*.csv
#
# Output Format
# -------------
# - Analysis: Console output of length distribution
# - Dataset: (N, 1152) feature matrix with DDG labels
#
# Usage Examples
# --------------
# 1. Analyze sequence lengths:
#    python sequence_alignment.py --analyze
#
# 2. Build training dataset:
#    python sequence_alignment.py --build
#
# =============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import dict, list, tuple
OUTPUT_DIR = "/data/store-data/yeyh/scripts/AF2stability/af2_output"
DATA_DIR = "/data/store-data/yeyh/scripts/AF2stability"
MAX_SEQ_LEN = 600


def load_representation(name: str) -> np.ndarray | None:
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
    path = os.path.join(OUTPUT_DIR, name, "single_representation.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)

def get_sequences_by_length() -> dict[int, list[str]]:
    """
    Group sequences by their length.
    Returns
    -------
    dict
        Mapping of length -> list of sequence names
    """
    if not os.path.exists(OUTPUT_DIR):
        return {}
    length_groups: dict[int, list[str]] = {}
    for name in os.listdir(OUTPUT_DIR):
        repr_path = os.path.join(OUTPUT_DIR, name, "single_representation.npy")
        if not os.path.exists(repr_path):
            continue
        repr = np.load(repr_path)
        L = repr.shape[0]
        if L not in length_groups:
            length_groups[L] = []
        length_groups[L].append(name)
    return length_groups

def analyze_sequence_lengths() -> dict[int, list[str]]:
    """
    Analyze and print sequence length distribution.
    Returns
    -------
    dict
        Length groups dictionary
    """
    length_groups = get_sequences_by_length()
    if not length_groups:
        print("No representations found!")
        return {}
    print("Sequence Length Distribution")
    print(f"Unique lengths: {len(length_groups)}")
    print(f"Min length: {min(length_groups.keys())}")
    print(f"Max length: {max(length_groups.keys())}")
    sorted_lengths = sorted(
        length_groups.items(), key=lambda x: len(x[1]), reverse=True
    )
    print(f"\n{'Length':<10} {'Count':<10}")
    print("-" * 25)
    for length, names in sorted_lengths[:10]:
        print(f"{length:<10} {len(names):<10}")
    if len(sorted_lengths) > 10:
        print(f"... and {len(sorted_lengths) - 10} more lengths")
    return length_groups

def extract_mutation_features(
    wt_repr: np.ndarray, mut_repr: np.ndarray, position: int
) -> np.ndarray | None:
    """
    Extract features at mutation position only.
    This is the simplest approach - use only the mutation position.
    No padding/truncation needed!
    Parameters
    ----------
    wt_repr : np.ndarray
        WT representation (L_wt, 384)
    mut_repr : np.ndarray
        Mut representation (L_mut, 384)
    position : int
        Mutation position (0-indexed)
    Returns
    -------
    np.ndarray or None
        1152-dimensional feature vector (384 × 3)
    """
    L_wt = wt_repr.shape[0]
    L_mut = mut_repr.shape[0]
    if position >= L_wt or position >= L_mut:
        return None
    wt_pos = wt_repr[position]
    mut_pos = mut_repr[position]
    diff = mut_pos - wt_pos
    features = np.concatenate([wt_pos, mut_pos, diff])
    return features

def build_training_dataset() -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Build training dataset with proper WT-Mut pairing.
    Returns
    -------
    tuple
        (X, y) where X is (N, 1152) and y is (N,)
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR, "fireprotdb_data_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "fireprotdb_data_test.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "fireprotdb_data_validation.csv"))
    all_df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    unique_seqs = all_df["sequence"].unique()
    seq_to_wt = {seq: idx for idx, seq in enumerate(unique_seqs)}
    features_list: list[np.ndarray] = []
    labels: list[float] = []
    valid_count = 0
    error_count = 0
    for idx, row in all_df.iterrows():
        seq = row["sequence"]
        position = row["position"] - 1
        wt_idx = seq_to_wt.get(seq, -1)
        if wt_idx < 0:
            error_count += 1
            continue
        wt_name = f"wt_{wt_idx}"
        mut_name = f"mut_{idx}"
        wt_repr = load_representation(wt_name)
        mut_repr = load_representation(mut_name)
        if wt_repr is None or mut_repr is None:
            error_count += 1
            continue
        features = extract_mutation_features(wt_repr, mut_repr, position)
        if features is not None:
            features_list.append(features)
            labels.append(row["ddg"])
            valid_count += 1
    print(f"Valid samples: {valid_count}")
    print(f"Missing representations: {error_count}")
    if not features_list:
        print("No valid samples found!")
        return None, None
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    # print(f"Dataset shape: X={X.shape}, y={y.shape}")
    return X, y




def main():
    parser = argparse.ArgumentParser(
        description="Handle sequence length alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --analyze    # Analyze sequence lengths
  %(prog)s --build      # Build training dataset
        """,
    )
    parser.add_argument(
        "--analyze", action="store_true", default=True, help="Analyze sequence lengths"
    )
    parser.add_argument("--build", action="store_true", help="Build training dataset")
    args = parser.parse_args()
    if args.build:
        build_training_dataset()
    else:
        analyze_sequence_lengths()





if __name__ == "__main__":
    main()
