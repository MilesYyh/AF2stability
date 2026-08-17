#!/usr/bin/env python
# '''
# @File    :
# @Time    :   2026/06/28 TianJin,China
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
VERSION = '1.0.0'


OUTPUT_FIELDNAMES = [
    'sample_id',
    'protein_id',
    'mutation_label',
    'position',
    'median_ddg',
    'wt_repr_path',
    'mut_repr_path',
    'wt_repr_shape',
    'mut_repr_shape',
    'feature_dim',
]

def resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    return repo_root / path_str

def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def write_summary(summary_json: Path, summary: dict[str, object]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract 1152-dim mutation features from WT/mutant single representations.'
    )
    parser.add_argument(
        '--input-manifest',
        default='data/manifests/fireprotdb_phase1_af2_smoketest_manifest.csv',
        help='Smoketest sample manifest CSV',
    )
    parser.add_argument(
        '--repr-output-dir',
        default='results/af2/single-repr-precomputed/outputs',
        help='Directory containing saved single_repr_*.npy files',
    )
    parser.add_argument(
        '--model-name',
        default='model_1_pred_0',
        help='Model name suffix for single representation files',
    )
    parser.add_argument(
        '--output-csv',
        default='data/processed/single_repr_features_smoketest.csv',
        help='Output feature metadata CSV',
    )
    parser.add_argument(
        '--output-npz',
        default='data/processed/single_repr_features_smoketest.npz',
        help='Output NPZ with feature matrix and labels',
    )
    parser.add_argument(
        '--summary-json',
        default='data/processed/single_repr_features_smoketest_summary.json',
        help='Output summary JSON',
    )
    parser.add_argument(
        '--strict-missing',
        action='store_true',
        help='Fail instead of skipping when WT or mutant single-representation files are missing',
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    manifest_path = resolve_repo_path(repo_root, args.input_manifest)
    repr_output_dir = resolve_repo_path(repo_root, args.repr_output_dir)
    output_csv = resolve_repo_path(repo_root, args.output_csv)
    output_npz = resolve_repo_path(repo_root, args.output_npz)
    summary_json = resolve_repo_path(repo_root, args.summary_json)

    rows_out: list[dict[str, str]] = []
    features: list[np.ndarray] = []
    ddg_values: list[float] = []
    sample_ids: list[str] = []
    total_rows = 0
    missing_wt_only = 0
    missing_mut_only = 0
    missing_both = 0
    missing_examples: list[dict[str, str]] = []

    wt_filename = f'single_repr_{args.model_name}.npy'

    for row in load_manifest_rows(manifest_path):
            total_rows += 1
            sample_id = row['sample_id']
            protein_id = row['protein_id']
            position = int(row['position'])
            mutation_label = row['mutation_label']
            ddg = float(row['median_ddg'])

            wt_path = repr_output_dir / protein_id / wt_filename
            mut_path = repr_output_dir / sample_id / wt_filename

            wt_exists = wt_path.exists()
            mut_exists = mut_path.exists()
            if not wt_exists or not mut_exists:
                if not wt_exists and not mut_exists:
                    missing_both += 1
                    missing_reason = 'missing_wt_and_mutant'
                elif not wt_exists:
                    missing_wt_only += 1
                    missing_reason = 'missing_wt'
                else:
                    missing_mut_only += 1
                    missing_reason = 'missing_mutant'

                if len(missing_examples) < 20:
                    missing_examples.append(
                        {
                            'sample_id': sample_id,
                            'protein_id': protein_id,
                            'reason': missing_reason,
                            'wt_repr_path': str(wt_path),
                            'mut_repr_path': str(mut_path),
                        }
                    )

                if args.strict_missing:
                    raise FileNotFoundError(
                        f'Missing representation for {sample_id}: '
                        f'wt_exists={wt_exists}, mut_exists={mut_exists}'
                    )
                continue

            wt_repr = np.load(wt_path)
            mut_repr = np.load(mut_path)

            residue_index = position - 1
            wt_vec = wt_repr[residue_index]
            mut_vec = mut_repr[residue_index]
            diff_vec = mut_vec - wt_vec
            feature_vec = np.concatenate([wt_vec, mut_vec, diff_vec], axis=0)

            if feature_vec.shape != (1152,):
                raise ValueError(
                    f'Unexpected feature shape for {sample_id}: {feature_vec.shape}'
                )

            rows_out.append(
                {
                    'sample_id': sample_id,
                    'protein_id': protein_id,
                    'mutation_label': mutation_label,
                    'position': str(position),
                    'median_ddg': row['median_ddg'],
                    'wt_repr_path': str(wt_path),
                    'mut_repr_path': str(mut_path),
                    'wt_repr_shape': str(tuple(wt_repr.shape)),
                    'mut_repr_shape': str(tuple(mut_repr.shape)),
                    'feature_dim': str(feature_vec.shape[0]),
                }
            )
            features.append(feature_vec)
            ddg_values.append(ddg)
            sample_ids.append(sample_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)

    feature_matrix = np.stack(features, axis=0) if features else np.zeros((0, 1152), dtype=np.float32)
    labels = np.array(ddg_values, dtype=np.float32)
    sample_id_array = np.array(sample_ids, dtype=object)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, X=feature_matrix, y=labels, sample_ids=sample_id_array)

    rows_missing_any = missing_wt_only + missing_mut_only + missing_both
    summary = {
        'version': VERSION,
        'input_manifest': str(manifest_path),
        'repr_output_dir': str(repr_output_dir),
        'model_name': args.model_name,
        'strict_missing': args.strict_missing,
        'manifest_rows': total_rows,
        'rows_extracted': len(rows_out),
        'rows_missing_any': rows_missing_any,
        'rows_missing_wt_only': missing_wt_only,
        'rows_missing_mut_only': missing_mut_only,
        'rows_missing_both': missing_both,
        'missing_examples': missing_examples,
        'feature_matrix_shape': list(feature_matrix.shape),
        'label_shape': list(labels.shape),
        'output_csv': str(output_csv),
        'output_npz': str(output_npz),
    }
    write_summary(summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))



if __name__ == '__main__':
    main()
