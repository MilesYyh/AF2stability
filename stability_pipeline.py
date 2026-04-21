#!/usr/bin/env python
# '''
# @File    :   
# @Time    :   2026/04/18 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO    :   
# """
# """
# AlphaFold3 Stability Change Prediction - Complete Pipeline
# ============================================================
# 
# This script implements a complete pipeline for predicting protein stability
# changes upon point mutations using AlphaFold3.
# 
# Based on: "Applications of AlphaFold beyond Protein Structure Prediction"
# (bioRxiv 2021, doi:10.1101/2021.11.03.467194)
# 
# Key Features:
# - Run AlphaFold3 predictions for WT and mutant sequences
# - Extract pLDDT, PAE features from outputs
# - Train regression model for ΔΔG prediction
# - Evaluate on test data
# 
# Usage:
#     python stability_pipeline.py --input_json INPUT --output_dir OUTPUT
# 
# """

import json
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import time
import hashlib
AF3_BASE_DIR = "/data/store-data/yeyh/tools/alphafold3"
AF3_SCRIPT = f"{AF3_BASE_DIR}/run_alphafold3.py"
AF3_MODELS_DIR = f"{AF3_BASE_DIR}/models"
AF3_DB_DIR = f"{AF3_BASE_DIR}/data_base"



class MutationData:
    """
    Container for mutation data
    """
    def __init__(
        self,
        protein_id: str,
        sequence: str,
        position: int,
        wild_type: str,
        mutant: str,
        ddg: float = None,
        dtm: float = None,
    ):
        self.protein_id = protein_id
        self.sequence = sequence
        self.position = position  # 0-indexed
        self.wild_type = wild_type.upper()
        self.mutant = mutant.upper()
        self.ddg = ddg
        self.dtm = dtm

    def get_mutant_sequence(self) -> str:
        seq_list = list(self.sequence)
        seq_list[self.position] = self.mutant
        return "".join(seq_list)
    @property
    def mutation_str(self) -> str:
        """String representation of mutation (e.g., V6G)"""
        return f"{self.wild_type}{self.position + 1}{self.mutant}"


class AF3Runner:
    """AlphaFold3 prediction runner"""
    def __init__(self, base_dir: str, models_dir: str, db_dir: str):
        self.base_dir = Path(base_dir)
        self.models_dir = Path(models_dir)
        self.db_dir = Path(db_dir)
        # Verify directories exist
        if not self.base_dir.exists():
            raise FileNotFoundError(f"AlphaFold3 base dir not found: {base_dir}")
        if not self.models_dir.exists():
            print(f"Warning: Models dir not found: {models_dir}")
        if not self.db_dir.exists():
            print(f"Warning: Database dir not found: {db_dir}")

    def create_input_json(self, name: str, sequence: str, chain_id: str = "A") -> Dict:
        """Create AlphaFold3 input JSON structure"""
        return {
            "name": name,
            "sequences": [{"protein": {"id": [chain_id], "sequence": sequence}}],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }

    def run_prediction(
        self,
        job_name: str,
        sequence: str,
        output_dir: str,
        gpu_id: int = 0,
        save_embeddings: bool = False,
        save_distogram: bool = False,
    ) -> Dict:
        """
        Run AlphaFold3 prediction for a sequence
        Returns path to output directory and confidence data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        input_json = self.create_input_json(job_name, sequence)
        json_path = output_path / f"{job_name}.json"
        with open(json_path, "w") as f:
            json.dump(input_json, f, indent=2)
        cmd = [
            "python",
            str(AF3_BASE_DIR / "run_alphafold3.py"),
            "--json_path",
            str(json_path),
            "--model_dir",
            str(self.models_dir),
            "--db_dir",
            str(self.db_dir),
            "--output_dir",
            str(output_path),
            "--gpu_device",
            str(gpu_id),
            "--run_data_pipeline",
            "true",
            "--run_inference",
            "true",
        ]
        if save_embeddings:
            cmd.append("--save_embeddings")
        if save_distogram:
            cmd.append("--save_distogram")

        print(f"Running AlphaFold3 for {job_name}...")
        print(f"  Sequence length: {len(sequence)}")
        try:
            # run AF3
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"  Error: {result.stderr[:500]}")
                return {"success": False, "error": result.stderr}

            # Load confidence data
            confidence_path = output_path / job_name / f"{job_name}_confidences.json"
            summary_path = (
                output_path / job_name / f"{job_name}_summary_confidences.json"
            )
            confidences = {}
            if confidence_path.exists():
                with open(confidence_path) as f:
                    confidences = json.load(f)
            summary = {}
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
            return {
                "success": True,
                "output_dir": str(output_path / job_name),
                "confidences": confidences,
                "summary": summary,
                "ranking_score": summary.get("ranking_score", 0.0),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout after 10 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class FeatureExtractor:
    """
    Extract features from AlphaFold3 predictions
    """

    @staticmethod
    def extract_from_confidences(confidences: Dict, seq_length: int) -> Dict:
        """Extract features from confidence JSON"""
        features = {
            "plddt_mean": 0.0,
            "plddt_at_mutation": 0.0,
            "plddt_min": 0.0,
            "plddt_max": 0.0,
            "pae_mean": 0.0,
            "pae_at_mutation": 0.0,
        }

        # Extract pLDDT 
        if "atom_plddts" in confidences and "atom_chain_ids" in confidences:
            atom_plddts = confidences["atom_plddts"]
            atom_chains = confidences["atom_chain_ids"]
            # Group by residue (simplified - assumes chain A)
            residue_plddts = []
            current_residue = None
            residue_scores = []

            for i, (plddt, chain) in enumerate(zip(atom_plddts, atom_chains)):
                if chain == "A":  # Only protein chain A
                    residue_scores.append(plddt)
                    if len(residue_scores) >= seq_length * 4:  # ~4 atoms per residue
                        break
            if residue_scores:
                # Average every ~4 atoms to get per-residue
                n_residues = min(len(residue_scores) // 4, seq_length)
                for i in range(n_residues):
                    start = i * 4
                    end = min((i + 1) * 4, len(residue_scores))
                    residue_plddts.append(np.mean(residue_scores[start:end]))
                features["plddt_mean"] = np.mean(residue_plddts)
                features["plddt_min"] = np.min(residue_plddts)
                features["plddt_max"] = np.max(residue_plddts)

        # Extract PAE
        if "pae" in confidences:
            pae = np.array(confidences["pae"])
            features["pae_mean"] = float(np.mean(pae)) if pae.size > 0 else 0.0
        return features

    @staticmethod
    def compute_mutation_features(
        wt_features: Dict, mut_features: Dict, mutation: MutationData
    ) -> np.ndarray:
        """
        Compute features for a mutation (WT vs Mutant comparison)
        """
        features = [
            # pLDDT differences
            wt_features.get("plddt_mean", 0.0),
            mut_features.get("plddt_mean", 0.0),
            mut_features.get("plddt_mean", 0.0) - wt_features.get("plddt_mean", 0.0),
            # pLDDT at mutation site (approximate - use nearby residues)
            wt_features.get("plddt_min", 0.0),
            mut_features.get("plddt_min", 0.0),
            # PAE differences
            wt_features.get("pae_mean", 0.0),
            mut_features.get("pae_mean", 0.0),
            # Amino acid properties
            len(mutation.sequence),  # sequence length
            mutation.position / len(mutation.sequence),  # relative position
            # AA property change (hydrophobicity)
            FeatureExtractor._get_aa_hydrophobicity(mutation.wild_type),
            FeatureExtractor._get_aa_hydrophobicity(mutation.mutant_aa),
            FeatureExtractor._get_aa_hydrophobicity(mutation.mutant_aa)
            - FeatureExtractor._get_aa_hydrophobicity(mutation.wild_type),
        ]
        return np.array(features)

    @staticmethod
    def _get_aa_hydrophobicity(aa: str) -> float:
        """Kyte-Doolittle hydrophobicity scale"""
        hydrophobicity = {
            "A": 1.8,
            "R": -4.5,
            "N": -3.5,
            "D": -3.5,
            "C": 2.5,
            "Q": -3.5,
            "E": -3.5,
            "G": -0.4,
            "H": -3.2,
            "I": 4.5,
            "L": 3.8,
            "K": -3.9,
            "M": 1.9,
            "F": 2.8,
            "P": -1.6,
            "S": -0.8,
            "T": -0.7,
            "W": -0.9,
            "Y": -1.3,
            "V": 4.2,
        }
        return hydrophobicity.get(aa.upper(), 0.0)


class StabilityModel:
    """ML model for predicting ΔΔG from features"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
        )
        self.model.fit(X_scaled, y)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="r2")
        print(f"  Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ΔΔG"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        y_pred = self.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "pearson": np.corrcoef(y, y_pred)[0, 1],
        }




def load_fireprotdb_data(filepath: str) -> List[MutationData]:
    # Expected format: protein_id, sequence, position, wild_type, mutant, ddg
    mutations = []
    if not os.path.exists(filepath):
        print(f"Warning: FireProtDB file not found: {filepath}")
        return mutations
    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        mut = MutationData(
            protein_id=row.get("protein_id", "unknown"),
            sequence=row["sequence"],
            position=int(row["position"]) - 1,  # Convert to 0-indexed
            wild_type=row["wild_type"],
            mutant=row["mutant"],
            ddg=float(row["ddg"]) if "ddg" in row else None,
        )
        mutations.append(mut)
    return mutations


def create_test_mutations() -> List[MutationData]:
    """Create small test mutation dataset"""
    test_proteins = [
        {
            "id": "test1",
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK",  # 70 aa
            "mutations": [
                (10, "A", "G", 0.5),
                (25, "V", "A", -0.8),
                (40, "K", "R", 1.2),
                (55, "L", "I", 0.3),
            ],
        },
        {
            "id": "test2",
            "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG",  # 70 aa
            "mutations": [
                (5, "V", "G", -0.5),
                (20, "A", "T", 0.8),
                (35, "L", "M", -1.2),
                (50, "F", "Y", 0.4),
            ],
        },
    ]
    mutations = []
    for protein in test_proteins:
        for pos, wt, mut, ddg in protein["mutations"]:
            mut_data = MutationData(
                protein_id=protein["id"],
                sequence=protein["sequence"],
                position=pos - 1,  # Convert to 0-indexed
                wild_type=wt,
                mutant=mut,
                ddg=ddg,
            )
            mutations.append(mut_data)
    return mutations


def run_pipeline(
    mutations: List[MutationData], output_dir: str, use_real_af3: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the complete stability prediction pipeline
    """
    af3_runner = AF3Runner(AF3_BASE_DIR, AF3_MODELS_DIR, AF3_DB_DIR)
    X_list = []
    y_list = []
    protein_mutations = {}
    for mut in mutations:
        if mut.protein_id not in protein_mutations:
            protein_mutations[mut.protein_id] = []
        protein_mutations[mut.protein_id].append(mut)
    print(f"\nProcessing {len(protein_mutations)} proteins...")
    for protein_id, muts in protein_mutations.items():
        wt_sequence = muts[0].sequence
        print(f"\n  Protein: {protein_id} (length: {len(wt_sequence)})")
        if use_real_af3:
            # Run WT prediction
            wt_result = af3_runner.run_prediction(
                job_name=f"{protein_id}_wt", sequence=wt_sequence, output_dir=output_dir
            )
            if not wt_result.get("success"):
                print(f"    Warning: WT prediction failed, using mock features")
                wt_features = {"plddt_mean": 75.0, "plddt_min": 50.0, "pae_mean": 5.0}
            else:
                wt_features = FeatureExtractor.extract_from_confidences(
                    wt_result["confidences"], len(wt_sequence)
                )
            # Run mutant predictions
            for mut in muts:
                mut_sequence = mut.get_mutant_sequence()
                mut_result = af3_runner.run_prediction(
                    job_name=f"{protein_id}_{mut.mutation_str}",
                    sequence=mut_sequence,
                    output_dir=output_dir,
                )
                if not mut_result.get("success"):
                    print(f"    Warning: Mutant prediction failed")
                    mut_features = {
                        "plddt_mean": 75.0,
                        "plddt_min": 50.0,
                        "pae_mean": 5.0,
                    }
                else:
                    mut_features = FeatureExtractor.extract_from_confidences(
                        mut_result["confidences"], len(mut_sequence)
                    )


                # Extract features
                features = FeatureExtractor.compute_mutation_features(
                    wt_features, mut_features, mut
                )
                X_list.append(features)
                if mut.ddg is not None:
                    y_list.append(mut.ddg)
        else:
            # Use mock features for testing
            for mut in muts:
                features = np.random.randn(12) * 2 + np.array(
                    [
                        75.0,
                        75.0,
                        0.0,
                        50.0,
                        50.0,
                        5.0,
                        5.0,
                        len(mut.sequence),
                        mut.position / len(mut.sequence),
                        1.0,
                        -1.0,
                        -2.0,
                    ]
                )
                X_list.append(features)
                if mut.ddg is not None:
                    y_list.append(mut.ddg)
    X = np.array(X_list)
    y = np.array(y_list)
    # print(f"\nCollected {len(X)} samples with {X.shape[1]} features")
    return X, y


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AlphaFold3-based protein stability change prediction"
    )
    parser.add_argument(
        "--output_dir",
        default="stability_output",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--use_real_af3",
        action="store_true",
        help="Run actual AlphaFold3 predictions (requires weights)",
    )
    parser.add_argument("--fireprotdb", default="", help="Path to FireProtDB CSV file")
    parser.add_argument("--test", action="store_true", help="Run with test data")
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    # Load mutations
    if args.test:
        mutations = create_test_mutations()
        print(f"Loaded {len(mutations)} test mutations")
    elif args.fireprotdb and os.path.exists(args.fireprotdb):
        mutations = load_fireprotdb_data(args.fireprotdb)
        print(f"Loaded {len(mutations)} mutations from FireProtDB")
    else:
        mutations = create_test_mutations()
        print(f"Using default test mutations ({len(mutations)} samples)")

    X, y = run_pipeline(mutations, args.output_dir, args.use_real_af3)
    model = StabilityModel()
    model.train(X, y)
    metrics = model.evaluate(X, y)
    print(f"  R² Score: {metrics['r2']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f} kcal/mol")
    print(f"  MAE: {metrics['mae']:.3f} kcal/mol")
    print(f"  Pearson Correlation: {metrics['pearson']:.3f}")






if __name__ == "__main__":
    main()
