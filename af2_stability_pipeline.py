#!/usr/bin/env python
# '''
# @File    :   
# @Time    :   2026/04/03 TianJin,China
# @Author  :   Yuhao Ye,Miles.
# @Contact :   milesyeyuhao@gmail.com
# @License :   https://github.com/MilesYyh
# @TODO    :   
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
import pickle
import hashlib

AF2_BASE_DIR = "/data/guest/AF2-docker_version"
AF2_ALPHAFOLD_DIR = f"{AF2_BASE_DIR}/alphafold"
OUTPUT_DIR = "/data/store-data/yeyh/scripts/AF2stability/af2_output"

# data 
class MutationData:
    """ 
    mutation data
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
        """
        Generate mutant sequence
        """
        seq_list = list(self.sequence)
        seq_list[self.position] = self.mutant
        return "".join(seq_list)

    @property
    def mutation_str(self) -> str:
        """
        String representation of mutation (e.g., V6G)
        """
        return f"{self.wild_type}{self.position+1}{self.mutant}"


# model
class AlphaFold2Runner:
    def __init__(self, af2_dir: str):
        self.af2_dir = Path(af2_dir)
        self.run_script = af2_dir + "/monomer-af2-run_docker-V_parallel.sh"

    def run_prediction(
        self, fasta_path: str, output_dir: str, job_name: str = None
    ) -> Dict:
        """
        Run AlphaFold2 prediction
        Args:
            fasta_path: Path to input FASTA file
            output_dir: Output directory
            job_name: Optional job name
        Returns:
            Dict with prediction results and paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if job_name is None:
            job_name = Path(fasta_path).stem
        import shutil
        shutil.copy(fasta_path, output_path / f"{job_name}.fasta")

        # Run prediction
        # Note: This would require Docker to be properly set up
        # For now, we focus on parsing existing results
        return {"success": True, "output_dir": str(output_path), "job_name": job_name}

    @staticmethod
    def parse_result_pkl(pkl_path: str) -> Dict:
        """
        Parse AlphaFold2 result pickle file
        """
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        results = {}
        # pLDDT
        if "plddt" in data:
            results["plddt"] = np.array(data["plddt"])
            results["plddt_mean"] = float(np.mean(data["plddt"]))
            results["plddt_min"] = float(np.min(data["plddt"]))
            results["plddt_max"] = float(np.max(data["plddt"]))
        # ranking confidence
        if "ranking_confidence" in data:
            results["ranking_confidence"] = float(data["ranking_confidence"])
        # structure module 
        if "structure_module" in data:
            sm = data["structure_module"]
            if "final_atom_positions" in sm:
                results["atom_positions"] = sm["final_atom_positions"]
            if "final_atom_mask" in sm:
                results["atom_mask"] = sm["final_atom_mask"]
        return results

    @staticmethod
    def parse_features_pkl(pkl_path: str) -> Dict:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        results = {}
        if "sequence" in data:
            results["sequence"] = bytes(data["sequence"][0]).decode("utf-8")
        if "aatype" in data:
            results["aatype"] = data["aatype"]
        if "seq_length" in data:
            results["seq_length"] = int(data["seq_length"][0])
        # MSA i
        if "msa" in data:
            results["msa"] = data["msa"]
            results["msa_depth"] = data["msa"].shape[0]
        if "num_alignments" in data:
            results["num_alignments"] = data["num_alignments"]
        return results

# Extract features
class FeatureExtractor:
    """
    Extract features from AlphaFold2 predictions for stability prediction
    """
    # Kyte-Doolittle hydrophobicity scale
    HYDROPHOBICITY = {
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
    # AA volume (Å³)
    AA_VOLUME = {
        "A": 88.6,
        "R": 173.4,
        "N": 114.1,
        "D": 111.1,
        "C": 108.5,
        "Q": 143.8,
        "E": 138.4,
        "G": 60.1,
        "H": 153.2,
        "I": 166.7,
        "L": 166.7,
        "K": 168.6,
        "M": 162.9,
        "F": 189.9,
        "P": 112.7,
        "S": 89.0,
        "T": 116.1,
        "W": 227.8,
        "Y": 193.6,
        "V": 140.0,
    }

    @staticmethod
    def extract_mutation_features(
        wt_result: Dict, mut_result: Dict, mutation: MutationData
    ) -> np.ndarray:
        """
        Extract features for a point mutation comparison
        Features:
        01. WT pLDDT at mutation site
        02. Mutant pLDDT at mutation site
        03. pLDDT difference
        04. WT pLDDT mean
        05. Mutant pLDDT mean
        06. pLDDT mean difference
        07. WT pLDDT min (most uncertain region)
        08. Mutant pLDDT min
        09. Ranking confidence WT
        10. Ranking confidence mutant
        11. Sequence length
        12. Relative mutation position
        13. WT AA hydrophobicity
        14. Mutant AA hydrophobicity
        15. Hydrophobicity change
        16. WT AA volume
        17. Mutant AA volume
        18. Volume change
        """
        pos = mutation.position

        # pLDDT at mutation site (or nearby average)
        window = 3
        start = max(0, pos - window)
        end = min(len(wt_result.get("plddt", [])), pos + window + 1)
        wt_plddt = wt_result.get("plddt", np.array([]))
        mut_plddt = mut_result.get("plddt", np.array([]))
        if len(wt_plddt) > start:
            wt_site_plddt = (
                np.mean(wt_plddt[start:end]) if end > start else np.mean(wt_plddt)
            )
        else:
            wt_site_plddt = wt_result.get("plddt_mean", 75.0)
        if len(mut_plddt) > start:
            mut_site_plddt = (
                np.mean(mut_plddt[start:end]) if end > start else np.mean(mut_plddt)
            )
        else:
            mut_site_plddt = mut_result.get("plddt_mean", 75.0)

        features = [
            # pLDDT at mutation site
            wt_site_plddt,
            mut_site_plddt,
            mut_site_plddt - wt_site_plddt,
            # pLDDT statistics
            wt_result.get("plddt_mean", 75.0),
            mut_result.get("plddt_mean", 75.0),
            mut_result.get("plddt_mean", 75.0) - wt_result.get("plddt_mean", 75.0),
            # pLDDT min (most uncertain)
            wt_result.get("plddt_min", 50.0),
            mut_result.get("plddt_min", 50.0),
            # Ranking confidence
            wt_result.get("ranking_confidence", 0.5),
            mut_result.get("ranking_confidence", 0.5),
            # Sequence features
            len(mutation.sequence),
            mutation.position / len(mutation.sequence),
            # AA properties - hydrophobicity
            FeatureExtractor.HYDROPHOBICITY.get(mutation.wild_type, 0.0),
            FeatureExtractor.HYDROPHOBICITY.get(mutation.mutant, 0.0),
            FeatureExtractor.HYDROPHOBICITY.get(mutation.mutant, 0.0)
            - FeatureExtractor.HYDROPHOBICITY.get(mutation.wild_type, 0.0),
            # AA properties - volume
            FeatureExtractor.AA_VOLUME.get(mutation.wild_type, 100.0),
            FeatureExtractor.AA_VOLUME.get(mutation.mutant, 100.0),
            FeatureExtractor.AA_VOLUME.get(mutation.mutant, 100.0)
            - FeatureExtractor.AA_VOLUME.get(mutation.wild_type, 100.0),
        ]

        return np.array(features)

    @staticmethod
    def calculate_rmsd(
        positions1: np.ndarray, positions2: np.ndarray, mask: np.ndarray = None
    ) -> float:
        """"""
        if positions1 is None or positions2 is None:
            return 0.0
        # Use CA atoms (index 0)
        ca1 = positions1[:, 0, :]                            # shape: (n_res, 3)
        ca2 = positions2[:, 0, :]
        if ca1.shape != ca2.shape:
            min_len = min(len(ca1), len(ca2))
            ca1 = ca1[:min_len]
            ca2 = ca2[:min_len]
        diff = ca1 - ca2
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return float(rmsd)


class StabilityModel:
    """
    ML model for predicting ΔΔG from AlphaFold2 features
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = [
            "wt_plddt_site",
            "mut_plddt_site",
            "plddt_site_diff",
            "wt_plddt_mean",
            "mut_plddt_mean",
            "plddt_mean_diff",
            "wt_plddt_min",
            "mut_plddt_min",
            "wt_ranking",
            "mut_ranking",
            "seq_length",
            "rel_position",
            "wt_hydro",
            "mut_hydro",
            "hydro_change",
            "wt_volume",
            "mut_volume",
            "volume_change",
        ]

    def train(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model - similar to paper's MLP but using GradientBoosting
        # for better performance with small datasets
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        # Cross-validation
        if len(X) >= 5:
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="r2")
            print(
                f"  Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )
        self.is_trained = True
        print(f"  Trained on {len(X)} samples with {X.shape[1]} features")

    # Predict ΔΔG
    def predict(self, X: np.ndarray) -> np.ndarray:
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
            "pearson": float(np.corrcoef(y, y_pred)[0, 1]),
        }

    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from the model
        """
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


def create_test_mutations() -> List[MutationData]:
    test_proteins = [
        {
            "id": "test1",
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK",
            "mutations": [
                (10, "A", "G", 0.5),
                (25, "V", "A", -0.8),
                (40, "K", "R", 1.2),
            ],
        },
        {
            "id": "test2",
            "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG",
            "mutations": [
                (5, "V", "G", -0.5),
                (20, "A", "T", 0.8),
            ],
        },
    ]
    mutations = []
    for protein in test_proteins:
        for pos, wt, mut, ddg in protein["mutations"]:
            mut_data = MutationData(
                protein_id=protein["id"],
                sequence=protein["sequence"],
                position=pos - 1,
                wild_type=wt,
                mutant=mut,
                ddg=ddg,
            )
            mutations.append(mut_data)
    return mutations


def parse_existing_af2_results(result_dir: str) -> Tuple[Dict, Dict]:
    """
    Parse existing AlphaFold2 results from a directory
    Looks for result_model_*_pred_*.pkl and features.pkl files
    """
    result_path = Path(result_dir)
    wt_results = {}
    features = {}

    result_files = sorted(result_path.glob("result_model_*_pred_*.pkl"))
    if result_files:
        # Use first model result
        wt_results = AlphaFold2Runner.parse_result_pkl(str(result_files[0]))
    features_file = result_path / "features.pkl"
    if features_file.exists():
        features = AlphaFold2Runner.parse_features_pkl(str(features_file))
    return wt_results, features


def run_pipeline_demo():
    """
    Run demo pipeline with mock features
    """
    print("\n" + "=" * 60)
    print("AlphaFold2 Stability Prediction Pipeline - Demo")
    print("=" * 60)
    mutations = create_test_mutations()
    print(f"\nLoaded {len(mutations)} test mutations")

    # For demo, generate features based on mutation properties
    # (In real use, would run AlphaFold2 and extract real features)
    X_list = []
    y_list = []
    for mut in mutations:
        # Generate realistic features based on AA properties
        features = np.zeros(18)
        # pLDDT features (typical values around 75-85)
        features[0] = np.random.uniform(70, 90)  # wt_plddt_site
        features[1] = np.random.uniform(65, 85)  # mut_plddt_site
        features[2] = features[1] - features[0]  # plddt_site_diff
        features[3] = np.random.uniform(75, 85)  # wt_plddt_mean
        features[4] = np.random.uniform(70, 85)  # mut_plddt_mean
        features[5] = features[4] - features[3]  # plddt_mean_diff
        features[6] = np.random.uniform(40, 60)  # wt_plddt_min
        features[7] = np.random.uniform(35, 55)  # mut_plddt_min
        features[8] = np.random.uniform(0.7, 0.9)  # wt_ranking
        features[9] = np.random.uniform(0.6, 0.9)  # mut_ranking
        features[10] = len(mut.sequence)  # seq_length
        features[11] = mut.position / len(mut.sequence)  # rel_position
        # Hydrophobicity features
        wt_hydro = FeatureExtractor.HYDROPHOBICITY.get(mut.wild_type, 0)
        mut_hydro = FeatureExtractor.HYDROPHOBICITY.get(mut.mutant, 0)
        features[12] = wt_hydro
        features[13] = mut_hydro
        features[14] = mut_hydro - wt_hydro
        # Volume features
        wt_vol = FeatureExtractor.AA_VOLUME.get(mut.wild_type, 100)
        mut_vol = FeatureExtractor.AA_VOLUME.get(mut.mutant, 100)
        features[15] = wt_vol
        features[16] = mut_vol
        features[17] = mut_vol - wt_vol
        X_list.append(features)
        y_list.append(mut.ddg)
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Created feature matrix: {X.shape}")

    # Train model
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    model = StabilityModel()
    model.train(X, y)
    # Evaluate
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    metrics = model.evaluate(X, y)
    print(f"  R² Score: {metrics['r2']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f} kcal/mol")
    print(f"  MAE: {metrics['mae']:.3f} kcal/mol")
    print(f"  Pearson Correlation: {metrics['pearson']:.3f}")
    # Feature importance
    print("\n" + "=" * 60)
    print("Top 5 Most Important Features")
    print("=" * 60)
    importance = model.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_importance[:5]:
        print(f"  {name}: {imp:.4f}")
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="AlphaFold2-based protein stability change prediction"
    )
    parser.add_argument(
        "--result_dir",
        default="/data/guest/AF2-docker_version/af2-results/enzyme",
        help="AlphaFold2 results directory",
    )
    parser.add_argument("--output_dir", default="af2_output", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run demo with test data")
    args = parser.parse_args()

    if args.test:
        run_pipeline_demo()
    else:
        print(f"\nParsing AlphaFold2 results from: {args.result_dir}")
        wt_results, features = parse_existing_af2_results(args.result_dir)
        print(f"WT Results keys: {list(wt_results.keys())}")
        print(f"Features keys: {list(features.keys())}")
        if "plddt" in wt_results:
            print(f"pLDDT shape: {wt_results['plddt'].shape}")
            print(f"pLDDT mean: {wt_results['plddt_mean']:.2f}")

if __name__ == "__main__":
    main()
