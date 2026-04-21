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
# File: extract_all_representations.py
# Description: Extract Single Representations from AlphaFold2 predictions
# =============================================================================
#
# Overview
# --------
# This script extracts the 384-dimensional single representations from trained
# AlphaFold2 model predictions. These representations are used as features for
# predicting protein stability changes (ΔΔG) from point mutations.
#
# Source: AlphaFold2 model forward pass (Evoformer output)
# Output: NumPy arrays of shape (L, 384) where L = sequence length
#
# =============================================================================
# | Parameter              | Type   | Default      | Description              |
# |------------------------|--------|--------------|--------------------------|
# | --max_samples          | int    | None         | Max samples to process   |
# | --test                 | flag   | False        | Test on 5 samples        |
# =============================================================================
#
# Input Requirements
# ------------------
# - AlphaFold2 prediction outputs in: /data/store-data/yeyh/scripts/AF2stability/af2_output/
# - Each subdirectory must contain: features.pkl
#
# Output Format
# -------------
# - single_representation.npy: (L, 384) float32 array per sequence
# - Saved to: af2_output/{name}/single_representation.npy
#
# Usage Examples
# --------------
# 1. Extract all representations:
#    python extract_all_representations.py
#
# 2. Test on first 5 samples:
#    python extract_all_representations.py --test
#
# 3. Limit to 100 samples:
#    python extract_all_representations.py --max_samples 100
#
# =============================================================================

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
AF2_DIR = "/data/store-data/yeyh/scripts/AF2stability/alphafold2"
sys.path.insert(0, AF2_DIR)
OUTPUT_DIR = "/data/store-data/yeyh/scripts/AF2stability/af2_output"
DATA_DIR = "/data/store-data/yeyh/scripts/AF2stability"


def load_features(name: str) -> dict | None:
    """
    Load features from AlphaFold2 output directory.
    Parameters
    ----------
    name : str
        Directory name (e.g., "wt_0", "mut_100")
    Returns
    -------
    dict or None
        Features dictionary if found, None otherwise
    """
    feat_path = os.path.join(OUTPUT_DIR, name, "features.pkl")
    if not os.path.exists(feat_path):
        return None
    with open(feat_path, "rb") as f:
        features = pickle.load(f)
    return features

def extract_single_representation(name: str, features: dict) -> np.ndarray:
    """
    Extract single representation from AlphaFold2 model.
    Parameters
    ----------
    name : str
        Sample name for logging
    features : dict
        Processed features from AlphaFold2
    Returns
    -------
    np.ndarray
        Single representation of shape (L, 384)
    """
    from alphafold.model import model, config, data
    model_config = config.model_config("model_1")
    model_config.data.eval.max_template_date = "2021-11-01"
    model_params = data.get_model_haiku_params("model_1", "/data/AFDB")
    model_runner = model.RunModel(model_config, model_params)

    # features
    processed_features = model_runner.process_features(features, random_seed=42)

    # Custom forward with return_representations
    import jax
    import jax.numpy as jnp
    import haiku as hk
    def _forward_fn(batch):
        from alphafold.model import modules
        af_model = modules.AlphaFold(model_config.model)
        return af_model(
            batch,
            is_training=False,
            compute_loss=False,
            ensemble_representations=True,
            return_representations=True,
        )
    apply_fn = jax.jit(hk.transform(_forward_fn).apply)
    result = apply_fn(model_params, jax.random.PRNGKey(42), processed_features)
    return result["representations"]["single"]

def get_mutation_position(
    wt_name: str, mut_name: str, data_dir: str = DATA_DIR
) -> tuple:
    """
    Find mutation position from training data.
    Parameters
    ----------
    wt_name : str
        Wild-type directory name
    mut_name : str
        Mutant directory name
    data_dir : str
        Data directory path
    Returns
    -------
    tuple
        (wt_sequence, mut_sequence, position, wt_aa, mut_aa)
    """
    df = pd.read_csv(os.path.join(data_dir, "fireprotdb_data_train.csv"))
    mut_idx = int(mut_name.replace("mut_", ""))
    if mut_idx < len(df):
        row = df.iloc[mut_idx]
        seq = row["sequence"]
        pos = row["position"] - 1  # Convert to 0-indexed
        wt_aa = row["wt_residue"]
        mut_aa = row["mut_residue"]
        return seq, seq[:pos] + mut_aa + seq[pos + 1 :], pos, wt_aa, mut_aa
    return None, None, None, None, None

def extract_for_mutation(
    wt_name: str, mut_name: str, data_dir: str = DATA_DIR
) -> dict | None:
    """
    Extract features for a mutation pair.
    Parameters
    ----------
    wt_name : str
        Wild-type name (e.g., "wt_0")
    mut_name : str
        Mutant name (e.g., "mut_0")
    data_dir : str
        Data directory
    Returns
    -------
    dict or None
        Dictionary with wt_repr, mut_repr, mutation_position, wt_aa, mut_aa
    """
    # WT features
    wt_features = load_features(wt_name)
    if wt_features is None:
        return None
    # Mut features
    mut_features = load_features(mut_name)
    if mut_features is None:
        return None
    # Extract representations
    print(f"Extracting {wt_name}...")
    wt_repr = extract_single_representation(wt_name, wt_features)
    np.save(os.path.join(OUTPUT_DIR, wt_name, "single_representation.npy"), wt_repr)
    print(f"Extracting {mut_name}...")
    mut_repr = extract_single_representation(mut_name, mut_features)
    np.save(os.path.join(OUTPUT_DIR, mut_name, "single_representation.npy"), mut_repr)

    # mutation position
    _, mut_seq, pos, wt_aa, mut_aa = get_mutation_position(wt_name, mut_name, data_dir)
    if pos is None:
        return None
    return {
        "wt_repr": wt_repr,
        "mut_repr": mut_repr,
        "mutation_position": pos,
        "wt_aa": wt_aa,
        "mut_aa": mut_aa,
    }

def batch_extract(max_samples: int | None = None) -> int:
    """
    Batch extract all representations.
    Parameters
    ----------
    max_samples : int, optional
        Maximum number of samples to process
    Returns
    -------
    int
        Number of successfully extracted representations
    """
    output_dirs = sorted(
        [
            d
            for d in os.listdir(OUTPUT_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DIR, d))
        ]
    )
    if max_samples:
        output_dirs = output_dirs[:max_samples]
    # print(f"Processing {len(output_dirs)} samples...")
    success = 0
    for name in output_dirs:
        repr_path = os.path.join(OUTPUT_DIR, name, "single_representation.npy")
        if os.path.exists(repr_path):
            print(f"Skip {name} - already done")
            continue
        # Load features
        features = load_features(name)
        if features is None:
            print(f"Skip {name} - no features")
            continue
        try:
            print(f"Processing {name}...")
            repr = extract_single_representation(name, features)
            np.save(repr_path, repr)
            success += 1
            print(f"Done: {repr.shape}")
        except Exception as e:
            print(f"Error: {e}")
    print(f"\nExtracted {success}/{len(output_dirs)} representations")
    return success

def main():
    parser = argparse.ArgumentParser(
        description="Extract Single Representations from AlphaFold2 predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Extract all representations
  %(prog)s --test               # Test on 5 samples
  %(prog)s --max_samples 100    # Process first 100 samples
        """,
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum samples to process"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test on small sample (5 sequences)"
    )
    args = parser.parse_args()
    if args.test:
        batch_extract(max_samples=5)
    else:
        batch_extract(max_samples=args.max_samples)



if __name__ == "__main__":
    main()


