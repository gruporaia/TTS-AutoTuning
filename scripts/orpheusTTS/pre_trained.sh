#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate aplicacao

python3 ../scripts/orpheusTTS/pre_trained.py "$1" "$2" "$3" "$4"

conda deactivate