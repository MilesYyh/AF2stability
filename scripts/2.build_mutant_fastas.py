#!/usr/bin/env python
# '''
# @File    :
# @Time    :   2026/06/10 TianJin,China
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
VERSION = '1.0.0'


def wrap_fasta_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i:i + width] for i in range(0, len(sequence), width))

def write_fasta(path: Path, header: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{header}\n{wrap_fasta_sequence(sequence)}\n", encoding="utf-8")

def resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    return repo_root / path_str

def load_clean_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_manifest_rows(
    clean_rows: list[dict[str, str]],
    wt_dir: Path,
    mutant_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, Path], dict[str, Path]]:
    manifest_rows: list[dict[str, str]] = []
    unique_wt_fastas: dict[str, Path] = {}
    unique_mutant_fastas: dict[str, Path] = {}

    for index, row in enumerate(clean_rows, start=1):
        protein_id = row["protein_id"]
        mutation_label = row["mutation_label"]
        wt_sequence = row["wild_type_sequence"]
        mutant_sequence = row["mutant_sequence"]
        mutation_id = f"{protein_id}__{mutation_label}"
        wt_path = wt_dir / f"{protein_id}.fasta"
        mutant_path = mutant_dir / f"{mutation_id}.fasta"

        if protein_id not in unique_wt_fastas:
            write_fasta(wt_path, protein_id, wt_sequence)
            unique_wt_fastas[protein_id] = wt_path

        write_fasta(mutant_path, mutation_id, mutant_sequence)
        unique_mutant_fastas[mutation_id] = mutant_path

        manifest_rows.append(
            {
                "sample_index": str(index),
                "sample_id": mutation_id,
                "protein_id": protein_id,
                "mutation_label": mutation_label,
                "position": row["position"],
                "wild_type": row["wild_type"],
                "mutation": row["mutation"],
                "median_ddg": row["median_ddg"],
                "num_experiments": row["num_experiments"],
                "sequence_length": row["sequence_length"],
                "wt_fasta_path": str(wt_path),
                "mutant_fasta_path": str(mutant_path),
                "wt_sequence": wt_sequence,
                "mutant_sequence": mutant_sequence,
            }
        )

    return manifest_rows, unique_wt_fastas, unique_mutant_fastas


def write_csv_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_summary(summary_json: Path, summary: dict[str, object]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

def build_summary(
    input_csv: Path,
    manifest_csv: Path,
    wt_dir: Path,
    mutant_dir: Path,
    manifest_rows: list[dict[str, str]],
    unique_wt_fastas: dict[str, Path],
    unique_mutant_fastas: dict[str, Path],
) -> dict[str, object]:
    return {
        "version": VERSION,
        "input_csv": str(input_csv),
        "manifest_csv": str(manifest_csv),
        "unique_wt_fastas": len(unique_wt_fastas),
        "unique_mutant_fastas": len(unique_mutant_fastas),
        "manifest_rows": len(manifest_rows),
        "wt_dir": str(wt_dir),
        "mutant_dir": str(mutant_dir),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WT/mutant FASTA files and AF2 manifest from the clean phase-1 dataset."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/fireprotdb_phase1_ddg_clean.csv",
        help="Clean phase-1 dataset CSV path",
    )
    parser.add_argument(
        "--wt-dir",
        default="data/interim/fasta/wild_type",
        help="Directory for WT FASTA files",
    )
    parser.add_argument(
        "--mutant-dir",
        default="data/interim/fasta/mutant",
        help="Directory for mutant FASTA files",
    )
    parser.add_argument(
        "--manifest-csv",
        default="data/manifests/fireprotdb_phase1_af2_manifest.csv",
        help="Output manifest CSV path",
    )
    parser.add_argument(
        "--summary-json",
        default="data/manifests/fireprotdb_phase1_af2_manifest_summary.json",
        help="Output manifest summary JSON path",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    input_csv = resolve_repo_path(repo_root, args.input_csv)
    wt_dir = resolve_repo_path(repo_root, args.wt_dir)
    mutant_dir = resolve_repo_path(repo_root, args.mutant_dir)
    manifest_csv = resolve_repo_path(repo_root, args.manifest_csv)
    summary_json = resolve_repo_path(repo_root, args.summary_json)

    clean_rows = load_clean_rows(input_csv)
    manifest_rows, unique_wt_fastas, unique_mutant_fastas = build_manifest_rows(
        clean_rows=clean_rows,
        wt_dir=wt_dir,
        mutant_dir=mutant_dir,
    )
    write_csv_rows(manifest_csv, manifest_rows)
    summary = build_summary(
        input_csv=input_csv,
        manifest_csv=manifest_csv,
        wt_dir=wt_dir,
        mutant_dir=mutant_dir,
        manifest_rows=manifest_rows,
        unique_wt_fastas=unique_wt_fastas,
        unique_mutant_fastas=unique_mutant_fastas,
    )
    write_summary(summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))




if __name__ == "__main__":
    main()
