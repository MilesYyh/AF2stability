#!/bin/bash
# run 4 instances in background with different GPUs

FASTA_DIR="/data/store-data/yeyh/scripts/AF2stability/fasta_input"
OUTPUT_DIR="/data/store-data/yeyh/scripts/AF2stability/af2_output"
RUN_SCRIPT="/data/guest/AF2-docker_version/alphafold/docker/run_docker.py"
files=($FASTA_DIR/*.fasta)
total=${#files[@]}

echo "Total files to process: $total"
run_gpu() {
    gpu=$1
    offset=$2
    echo "GPU $job starting from offset $offset"
    for i in $(seq $offset 4 $((total-1))); do
        fasta="${files[$i]}"
        name=$(basename ${fasta%.fasta})
        outdir=$OUTPUT_DIR/$name
        if [ -f "$outdir/ranked_0.pdb" ]; then
            echo "[GPU$gpu] Skip $name (done)"
            continue
        fi
        echo "[GPU$gpu] Running $name..."
        python3 $RUN_SCRIPT \
            --fasta_paths=$fasta \
            --data_dir=/data/AFDB \
            --output_dir=$OUTPUT_DIR \
            --model_preset=monomer \
            --max_template_date=2021-11-01 \
            --gpu_devices=$gpu \
            > /tmp/af2_${name}.log 2>&1
        if [ -f "$outdir/ranked_0.pdb" ]; then
            echo "[GPU$gpu] Done $name"
        else
            echo "[GPU$gpu] Failed $name"
        fi
    done
}


run_gpu 0 0 &
run_gpu 1 1 &
run_gpu 2 2 &
run_gpu 3 3 &
echo "'docker ps' to monitor, 'ls $OUTPUT_DIR' to see progress"