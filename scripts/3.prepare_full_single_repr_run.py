#!/usr/bin/env python
# '''
# @File    :
# @Time    :   2026/06/13 TianJin,China
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


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"

def resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    return repo_root / path_str

def load_manifest_rows(input_manifest: Path) -> list[dict[str, str]]:
    with input_manifest.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def build_tasks(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], set[str]]:
    tasks: list[dict[str, str]] = []
    seen_wt: set[str] = set()
    for row in rows:
        protein_id = row['protein_id']
        sample_id = row['sample_id']
        wt_fasta = row['wt_fasta_path']
        mut_fasta = row['mutant_fasta_path']
        if protein_id not in seen_wt:
            tasks.append(
                {
                    'task_id': f'{sample_id}__wt',
                    'fasta_path': wt_fasta,
                    'fasta_name': Path(wt_fasta).stem,
                }
            )
            seen_wt.add(protein_id)
        tasks.append(
            {
                'task_id': f'{sample_id}__mutant',
                'fasta_path': mut_fasta,
                'fasta_name': Path(mut_fasta).stem,
            }
        )
    return tasks, seen_wt

def write_summary(summary_json: Path, summary: dict[str, object]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

def build_summary(
    *,
    input_manifest: Path,
    rows: list[dict[str, str]],
    seen_wt: set[str],
    tasks: list[dict[str, str]],
    gpu_ids: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    log_dir: Path,
    runner_script: Path,
    run_script: Path,
    custom_run_alphafold: Path,
) -> dict[str, object]:
    return {
        'version': VERSION,
        'input_manifest': str(input_manifest),
        'total_rows': len(rows),
        'unique_wt_sequences': len(seen_wt),
        'total_tasks': len(tasks),
        'gpu_ids': gpu_ids,
        'data_dir': args.data_dir,
        'output_dir': str(output_dir),
        'log_dir': str(log_dir),
        'runner_script': str(runner_script),
        'run_script': str(run_script),
        'run_alphafold_source_path': str(custom_run_alphafold),
        'save_single_representation': True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare a full 4-GPU AlphaFold single-representation run for the clean dataset.'
    )
    parser.add_argument(
        '--input-manifest',
        default='data/manifests/fireprotdb_phase1_af2_manifest.csv',
        help='Full AF2 manifest CSV',
    )
    parser.add_argument(
        '--gpu-ids',
        default='0,1,2,3',
        help='Comma-separated GPU ids',
    )
    parser.add_argument(
        '--data-dir',
        default='/data/AFDB',
        help='AlphaFold database directory',
    )
    parser.add_argument(
        '--docker-image-name',
        default='alphafold',
        help='Docker image name',
    )
    parser.add_argument(
        '--max-template-date',
        default='2021-11-01',
        help='Max template date',
    )
    parser.add_argument(
        '--model-preset',
        default='monomer',
        choices=['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
        help='AlphaFold model preset',
    )
    parser.add_argument(
        '--output-dir',
        default='results/af2/single-repr-full/outputs',
        help='Output directory for full single-repr run',
    )
    parser.add_argument(
        '--log-dir',
        default='results/af2/single-repr-full/logs',
        help='Log directory for full single-repr run',
    )
    parser.add_argument(
        '--runner-script',
        default='results/af2/single-repr-full/run_single_repr_full.sh',
        help='Generated bash runner path',
    )
    parser.add_argument(
        '--summary-json',
        default='results/af2/single-repr-full/summary.json',
        help='Summary JSON path',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    input_manifest = resolve_repo_path(repo_root, args.input_manifest)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    log_dir = resolve_repo_path(repo_root, args.log_dir)
    runner_script = resolve_repo_path(repo_root, args.runner_script)
    summary_json = resolve_repo_path(repo_root, args.summary_json)
    run_script = resolve_repo_path(repo_root, 'alphafold2/docker/run_docker.py')
    custom_run_alphafold = resolve_repo_path(repo_root, 'alphafold2/run_alphafold.py')

    rows = load_manifest_rows(input_manifest)
    tasks, seen_wt = build_tasks(rows)

    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(',') if gpu.strip()]
    if len(gpu_ids) != 4:
        raise ValueError(f'Expected exactly 4 GPU ids, got: {gpu_ids}')

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_script.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        '',
        f"DATA_DIR={shell_quote(args.data_dir)}",
        f"RUN_SCRIPT={shell_quote(str(run_script))}",
        f"OUTPUT_DIR={shell_quote(str(output_dir))}",
        f"LOG_DIR={shell_quote(str(log_dir))}",
        f"DOCKER_IMAGE_NAME={shell_quote(args.docker_image_name)}",
        f"MAX_TEMPLATE_DATE={shell_quote(args.max_template_date)}",
        f"MODEL_PRESET={shell_quote(args.model_preset)}",
        'mkdir -p "$OUTPUT_DIR" "$LOG_DIR"',
        '',
        f'TASK_COUNT={len(tasks)}',
        f'GPU_COUNT={len(gpu_ids)}',
        '',
    ]

    for idx, task in enumerate(tasks):
        lines.append(f"FASTA_{idx}={shell_quote(task['fasta_path'])}")
        lines.append(f"NAME_{idx}={shell_quote(task['fasta_name'])}")
        lines.append(f"TASKID_{idx}={shell_quote(task['task_id'])}")
    lines.append('')

    lines.extend([
        'run_gpu() {',
        '  local gpu=$1',
        '  local offset=$2',
        '  echo "[GPU${gpu}] starting from offset ${offset}"',
        '  for ((i=offset; i<TASK_COUNT; i+=GPU_COUNT)); do',
        '    local fasta_var=FASTA_${i}',
        '    local name_var=NAME_${i}',
        '    local taskid_var=TASKID_${i}',
        '    local fasta=${!fasta_var}',
        '    local name=${!name_var}',
        '    local taskid=${!taskid_var}',
        '    local outdir="$OUTPUT_DIR/$name"',
        '    local logfile="$LOG_DIR/${taskid}.log"',
        '    if [ -f "$outdir/single_repr_model_1_pred_0.npy" ]; then',
        '      echo "[GPU${gpu}] Skip ${name} (single repr already dumped)"',
        '      continue',
        '    fi',
        '    echo "[GPU${gpu}] Running ${name} with single repr dump -> ${logfile}"',
        '    python3 "$RUN_SCRIPT" \\',
        '      --use_gpu=true \\',
        '      --gpu_devices="$gpu" \\',
        '      --save_single_representation=true \\',
        f'      --run_alphafold_source_path={shell_quote(str(custom_run_alphafold))} \\',
        '      --fasta_paths="$fasta" \\',
        '      --data_dir="$DATA_DIR" \\',
        '      --output_dir="$OUTPUT_DIR" \\',
        '      --docker_image_name="$DOCKER_IMAGE_NAME" \\',
        '      --model_preset="$MODEL_PRESET" \\',
        '      --max_template_date="$MAX_TEMPLATE_DATE" \\',
        '      > "$logfile" 2>&1',
        '  done',
        '}',
        '',
    ])

    for offset, gpu in enumerate(gpu_ids):
        lines.append(f"run_gpu {shell_quote(gpu)} {offset} &")
    lines.extend([
        'wait',
        'echo "Full single-representation run finished."',
        '',
    ])

    runner_script.write_text('\n'.join(lines), encoding='utf-8')
    runner_script.chmod(0o755)

    summary = build_summary(
        input_manifest=input_manifest,
        rows=rows,
        seen_wt=seen_wt,
        tasks=tasks,
        gpu_ids=gpu_ids,
        args=args,
        output_dir=output_dir,
        log_dir=log_dir,
        runner_script=runner_script,
        run_script=run_script,
        custom_run_alphafold=custom_run_alphafold,
    )
    write_summary(summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))




if __name__ == '__main__':
    main()
