#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate xtts

python3 ../scripts/xTTS-v2/finetune.py "$1" "$2" "$3" "$4" "$5"

conda deactivate