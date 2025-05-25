#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate aplicacao

python3 ../scripts/xTTS-v2/synthesize.py "$1" "$2" "$3"

conda deactivate