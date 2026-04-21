#!/usr/bin/env python
# '''
# @File    :   
# @Time    :   2026/04/21 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO    :   
# """

# Prepare FireProtDB data for AlphaFold2 stability prediction
# This script processes the FireProtDB dataset and prepares it for training
# the stability prediction model.

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_fireprotdb(
    train_path: str, test_path: str, val_path: str = None
) -> pd.DataFrame:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    if val_path:
        val = pd.read_parquet(val_path)
        df = pd.concat([train, test, val], ignore_index=True)
    else:
        df = pd.concat([train, test], ignore_index=True)
    return df


def filter_valid_mutations(
    df: pd.DataFrame,
    min_seq_length: int = 30,
    max_seq_length: int = 600,
    ddg_range: tuple = (-10, 10),
) -> pd.DataFrame:
    """
    Filter for valid single-point mutations with complete data
    Args:
        df: Input dataframe
        min_seq_length: Minimum sequence length
        max_seq_length: Maximum sequence length
        ddg_range: Valid DDG range (kcal/mol)
    Returns:
        Filtered dataframe
    """
    # Filter for valid DDG values
    df = df[df["ddg"].notna()].copy()
    df = df[(df["ddg"] >= ddg_range[0]) & (df["ddg"] <= ddg_range[1])]
    df = df[df["sequence"].notna()].copy()
    df = df[df["sequence"].str.len() >= min_seq_length]
    df = df[df["sequence"].str.len() <= max_seq_length]
    df = df[df["wt_residue"].notna()].copy()
    df = df[df["mut_residue"].notna()].copy()
    df = df[df["position"].notna()].copy()
    # position is within sequence bounds
    df = df[df["position"] <= df["sequence"].str.len()]
    return df


def prepare_mutation_data(
    df: pd.DataFrame, max_mutations_per_protein: int = None, deduplicate: bool = True
) -> pd.DataFrame:
    """
    Prepare mutation data for model training
    Args:
        df: Filtered dataframe
        max_mutations_per_protein: Max mutations per protein (for balancing)
        deduplicate: Remove duplicate mutations (same protein, position, mutation)
    Returns:
        Prepared dataframe
    """
    if deduplicate:
        df = df.drop_duplicates(
            subset=["protein_id", "position", "mutation"], keep="first"
        )
    if max_mutations_per_protein and max_mutations_per_protein > 0:
        df = df.groupby("protein_id").head(max_mutations_per_protein)
    return df


def create_training_data(
    df: pd.DataFrame,
    output_path: str,
    include_pdb: bool = True,
    split_ratio: tuple = (0.8, 0.1, 0.1),
) -> dict:
    """
    Create training/validation/test splits
    Args:
        df: Prepared dataframe
        output_path: Output CSV path
        include_pdb: Only include entries with PDB IDs
        split_ratio: Train/Val/Test ratio
    Returns:
        Statistics dictionary
    """
    if include_pdb:
        df = df[df["pdb_id"].notna()].copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_size = int(n * split_ratio[0])
    val_size = int(n * split_ratio[1])
    train_df = df[:train_size]
    val_df = df[train_size : train_size + val_size]
    test_df = df[train_size + val_size :]
    essential_cols = [
        "protein_id",
        "pdb_id",
        "sequence",
        "wt_residue",
        "position",
        "mut_residue",
        "mutation",
        "ddg",
        "dtm",
        "tm",
    ]
    available_cols = [c for c in essential_cols if c in df.columns]
    train_df[available_cols].to_csv(f"{output_path}_train.csv", index=False)
    val_df[available_cols].to_csv(f"{output_path}_validation.csv", index=False)
    test_df[available_cols].to_csv(f"{output_path}_test.csv", index=False)

    stats = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "unique_proteins": df["protein_id"].nunique(),
        "ddg_mean": df["ddg"].mean(),
        "ddg_std": df["ddg"].std(),
        "ddg_min": df["ddg"].min(),
        "ddg_max": df["ddg"].max(),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare FireProtDB data")
    parser.add_argument(
        "--train",
        default="/data/store-data/yeyh/scripts/AF2stability/train.parquet",
        help="Training data parquet file",
    )
    parser.add_argument(
        "--test",
        default="/data/store-data/yeyh/scripts/AF2stability/test.parquet",
        help="Test data parquet file",
    )
    parser.add_argument(
        "--validation",
        default="/data/store-data/yeyh/scripts/AF2stability/validation.parquet",
        help="Validation data parquet file",
    )
    parser.add_argument(
        "--output",
        default="/data/store-data/yeyh/scripts/AF2stability/fireprotdb_data",
        help="Output base path (without extension)",
    )
    parser.add_argument(
        "--min_seq_len", type=int, default=30, help="Minimum sequence length"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=600, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_per_protein", type=int, default=50, help="Maximum mutations per protein"
    )
    parser.add_argument(
        "--with_pdb_only", action="store_true", help="Only include entries with PDB IDs"
    )
    args = parser.parse_args()


    df = load_fireprotdb(args.train, args.test, args.validation)
    df = filter_valid_mutations(
        df, min_seq_length=args.min_seq_len, max_seq_length=args.max_seq_len
    )
    df = prepare_mutation_data(df, max_mutations_per_protein=args.max_per_protein)
    stats = create_training_data(df, args.output, include_pdb=args.with_pdb_only)

    print(stats)





if __name__ == "__main__":
    main()
