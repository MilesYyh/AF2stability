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
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
class MutationRecord(TypedDict):
    experiment_id: str
    protein_name: str
    protein_id: str
    protein_id_source: str
    uniprot_id: str
    raw_pdb_id: str
    normalized_pdb_id: str
    chain: str
    position: int
    wild_type: str
    mutation: str
    ddg: float
    dTm: str
    sequence: str
    sequence_hash: str
    dataset_tags: str
    is_curated: str
    publication_doi: str
    publication_pubmed: str


def apply_mutation(sequence: str, position: int, mutation: str) -> str:
    index = position - 1
    return f"{sequence[:index]}{mutation}{sequence[index + 1:]}"
def clean_str(value: str | None) -> str:
    return (value or "").strip()
def normalize_sequence(sequence: str) -> str:
    return clean_str(sequence).upper().replace(" ", "")
def normalize_pdb_id(raw_pdb_id: str) -> str:
    pdb_id = clean_str(raw_pdb_id)
    if not pdb_id:
        return ""
    return pdb_id.split("|")[0].upper()
def is_standard_residue(value: str) -> bool:
    return len(value) == 1 and value in STANDARD_AA
def make_sequence_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean phase-1 FireProtDB ΔΔG dataset."
    )
    parser.add_argument(
        "--input",
        default="dataset/fireprotdb_results.csv",
        help="Input FireProtDB CSV path",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/fireprotdb_phase1_ddg_clean.csv",
        help="Output clean mutation table CSV path",
    )
    parser.add_argument(
        "--output-summary",
        default="data/processed/fireprotdb_phase1_ddg_summary.json",
        help="Output summary JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)

    grouped_rows: dict[tuple[str, int, str, str], list[MutationRecord]] = defaultdict(list)
    summary = Counter()
    exclusion_reasons = Counter()
    missing_identifier_counter = Counter()

    unique_protein_ids = set()
    unique_sequences = set()

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary["total_rows"] += 1
            experiment_id = clean_str(row.get("experiment_id"))
            uniprot_id = clean_str(row.get("uniprot_id")).upper()
            raw_pdb_id = clean_str(row.get("pdb_id"))
            normalized_pdb_id = normalize_pdb_id(raw_pdb_id)
            chain = clean_str(row.get("chain")).upper()
            sequence = normalize_sequence(row.get("sequence", ""))
            wild_type = clean_str(row.get("wild_type")).upper()
            mutation = clean_str(row.get("mutation")).upper()
            position_raw = clean_str(row.get("position"))
            ddg_raw = clean_str(row.get("ddG"))
            if not ddg_raw:
                exclusion_reasons["missing_ddg"] += 1
                continue
            summary["rows_with_ddg"] += 1

            try:
                ddg = float(ddg_raw)
                if math.isnan(ddg) or math.isinf(ddg):
                    raise ValueError
            except ValueError:
                exclusion_reasons["invalid_ddg"] += 1
                continue

            if not sequence:
                exclusion_reasons["missing_sequence"] += 1
                continue
            if any(residue not in STANDARD_AA for residue in sequence):
                exclusion_reasons["non_standard_sequence"] += 1
                continue
            if not position_raw:
                exclusion_reasons["missing_position"] += 1
                continue

            try:
                position = int(position_raw)
            except ValueError:
                exclusion_reasons["invalid_position"] += 1
                continue

            if position < 1 or position > len(sequence):
                exclusion_reasons["position_out_of_range"] += 1
                continue
            if not is_standard_residue(wild_type):
                exclusion_reasons["invalid_wild_type"] += 1
                continue
            if not is_standard_residue(mutation):
                exclusion_reasons["invalid_mutation"] += 1
                continue
            if wild_type == mutation:
                exclusion_reasons["same_wt_and_mutation"] += 1
                continue
            residue_at_position = sequence[position - 1]
            if residue_at_position != wild_type:
                exclusion_reasons["wild_type_sequence_mismatch"] += 1
                continue

            protein_id = ""
            protein_id_source = ""
            if uniprot_id:
                protein_id = uniprot_id
                protein_id_source = "uniprot_id"
            elif normalized_pdb_id and chain:
                protein_id = f"{normalized_pdb_id}_{chain}"
                protein_id_source = "pdb_id_chain"
            else:
                if not uniprot_id:
                    missing_identifier_counter["missing_uniprot_id"] += 1
                if not normalized_pdb_id:
                    missing_identifier_counter["missing_pdb_id"] += 1
                if not chain:
                    missing_identifier_counter["missing_chain"] += 1
                exclusion_reasons["missing_protein_identifier"] += 1
                continue

            summary["rows_after_phase1_filters"] += 1
            unique_protein_ids.add(protein_id)
            unique_sequences.add(sequence)

            mutation_key = (protein_id, position, wild_type, mutation)
            record: MutationRecord = {
                "experiment_id": experiment_id,
                "protein_name": clean_str(row.get("protein_name")),
                "protein_id": protein_id,
                "protein_id_source": protein_id_source,
                "uniprot_id": uniprot_id,
                "raw_pdb_id": raw_pdb_id,
                "normalized_pdb_id": normalized_pdb_id,
                "chain": chain,
                "position": position,
                "wild_type": wild_type,
                "mutation": mutation,
                "ddg": ddg,
                "dTm": clean_str(row.get("dTm")),
                "sequence": sequence,
                "sequence_hash": make_sequence_hash(sequence),
                "dataset_tags": clean_str(row.get("datasets")),
                "is_curated": clean_str(row.get("is_curated")),
                "publication_doi": clean_str(row.get("publication_doi")),
                "publication_pubmed": clean_str(row.get("publication_pubmed")),
            }
            grouped_rows[mutation_key].append(record)

    output_rows: list[dict[str, object]] = []
    duplicate_group_sizes = Counter()

    for mutation_key in sorted(grouped_rows):
        records = grouped_rows[mutation_key]
        duplicate_group_sizes[len(records)] += 1
        ddg_values: list[float] = [record["ddg"] for record in records]
        median_ddg = statistics.median(ddg_values)
        representative = records[0]
        wt_sequence = representative["sequence"]
        mutant_sequence = apply_mutation(
            wt_sequence,
            representative["position"],
            representative["mutation"],
        )
        output_rows.append(
            {
                "protein_id": representative["protein_id"],
                "protein_id_source": representative["protein_id_source"],
                "uniprot_id": representative["uniprot_id"],
                "pdb_id": representative["normalized_pdb_id"],
                "raw_pdb_id": representative["raw_pdb_id"],
                "chain": representative["chain"],
                "protein_name": representative["protein_name"],
                "position": representative["position"],
                "wild_type": representative["wild_type"],
                "mutation": representative["mutation"],
                "mutation_label": (
                    f"{representative['wild_type']}{representative['position']}{representative['mutation']}"
                ),
                "median_ddg": f"{median_ddg:.6f}",
                "num_experiments": len(records),
                "sequence_length": len(wt_sequence),
                "wild_type_sequence": wt_sequence,
                "mutant_sequence": mutant_sequence,
                "sequence_hash": representative["sequence_hash"],
                "dataset_tags": "|".join(
                    sorted({record["dataset_tags"] for record in records if record["dataset_tags"]})
                ),
                "experiment_ids": "|".join(record["experiment_id"] for record in records if record["experiment_id"]),
                "ddg_values": "|".join(str(record["ddg"]) for record in records),
                "publication_dois": "|".join(
                    sorted({record["publication_doi"] for record in records if record["publication_doi"]})
                ),
                "publication_pubmeds": "|".join(
                    sorted({record["publication_pubmed"] for record in records if record["publication_pubmed"]})
                ),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(output_rows[0].keys()) if output_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary_payload = {
        "input_csv": str(input_path),
        "output_csv": str(output_csv),
        "total_rows": summary["total_rows"],
        "rows_with_ddg": summary["rows_with_ddg"],
        "rows_after_phase1_filters": summary["rows_after_phase1_filters"],
        "unique_mutations_after_collapse": len(output_rows),
        "unique_protein_ids": len(unique_protein_ids),
        "unique_sequences": len(unique_sequences),
        "rows_removed_by_duplicate_collapse": summary["rows_after_phase1_filters"] - len(output_rows),
        "duplicate_group_size_distribution": dict(sorted(duplicate_group_sizes.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "missing_identifier_counts_for_excluded_rows": dict(sorted(missing_identifier_counter.items())),
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))





if __name__ == "__main__":
    main()
