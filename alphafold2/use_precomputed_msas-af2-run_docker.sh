#!/bin/bash

DATA_DIR="/data/AFDB"
OUTPUT_DIR="/home/lish/AF2-docker_version/af2-results/"
for seqid in T_0.1-sample_2-score_0.8567-global_score_0.9148-seq_recovery_0.4473 T_0.1-sample_3-score_0.8729-global_score_0.9341-seq_recovery_0.4530 T_0.1-sample_1-score_0.8511-global_score_0.9147-seq_recovery_0.4473 ; do
    echo "--------------------------------"
    echo "$seqid is going to processing"
    echo "--------------------------------"
    nohup python ../alphafold/docker/run_docker.py \
	--fasta_paths=$seqid \
	--data_dir="$DATA_DIR" --output_dir="$OUTPUT_DIR" \
	--model_preset=monomer \
	--max_template_date=2021-11-01\
	--use_precomputed_msas=true\
	 > "$OUTPUT_DIR/$seqid.out" & \
done
