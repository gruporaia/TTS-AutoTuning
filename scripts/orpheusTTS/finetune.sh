#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate unsloth_env

CUDA_VISIBLE_DEVICES=0 python ../scripts/orpheusTTS/train_tuning.py "$1" "$2" "$3" "$4" "$5"

conda deactivate