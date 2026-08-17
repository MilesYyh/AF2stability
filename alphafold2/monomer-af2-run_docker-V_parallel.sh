#!/bin/bash

echo T_0.1-sample_2-score_0.8567-global_score_0.9148-seq_recovery_0.4473 T_0.1-sample_3-score_0.8729-global_score_0.9341-seq_recovery_0.4530 T_0.1-sample_1-score_0.8511-global_score_0.9147-seq_recovery_0.4473 | xargs -n 1 | xargs -P 16 -I {} sh -c 'nohup python ../alphafold/docker/run_docker.py --fasta_paths="{}" --data_dir=/data/AFDB --output_dir=/home/lish/AF2-docker_version/af2-results/ --model_preset=monomer --max_template_date=2021-11-01 --use_precomputed_msas=False > /home/lish/AF2-docker_version/af2-results/{}.out &'
