#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate unsloth_env

python3 ../scripts/orpheusTTS/inference_tuning.py "$1" "$2" "$3"

conda deactivate