#!/bin/bash

echo N10.fasta N9.fasta ISvNY2A_2  SCe-1-EGA80285_1-  VCa-2-partiallowercase | xargs -n 1 | xargs -P 5 -I {} sh -c 'nohup python ../alphafold/docker/run_docker.py --fasta_paths="{}" --max_template_date=2021-11-01 --model_preset=multimer --data_dir=/data/AFDB --output_dir=/home/lish/AF2-docker_version/af2-results/ > /home/lish/AF2-docker_version/af2-results/{}.out &'
