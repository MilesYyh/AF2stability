#!/bin/bash

DATA_DIR="/data/AFDB"
OUTPUT_DIR="/data/guest/AF2-docker_version/af2-results/"
for seqid in enzyme.fas ; do
    echo "--------------------------------"
    echo "$seqid is going to processing"
    echo "--------------------------------"
    nohup python ../alphafold/docker/run_docker.py \
	--fasta_paths=$seqid \
	--data_dir="$DATA_DIR" --output_dir="$OUTPUT_DIR" \
	--model_preset=monomer \
	--max_template_date=2021-11-01\
	 > "$OUTPUT_DIR/$seqid.out" & \
done
