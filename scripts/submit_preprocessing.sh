#!/bin/bash
#SBATCH --job-name=run_preprocessing
#SBATCH --output=logs/run_preprocessing_%j.log  
#SBATCH --error=logs/run_preprocessing_error_%j.log             
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1              
#SBATCH --time=12:00:00                 
#SBATCH --partition=normal   
#SBATCH --constraint=gpu


# 2) Load needed modules in correct order:
module load cray/23.12
module load gcc-native/12.3
module load cray-python/3.11.5
module load cudatoolkit/24.3_12.3
# Line removed by filter-repo due to potential secret
# Line removed by filter-repo due to potential secret
# 6) Run Python script
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision="fp16" \
  --num_machines=1 \
  --dynamo_backend="no" \
  ../run_preprocessing.py \
    --dataset_path data/cs_bulletin_ocr_en_1009.csv \
    --output_dir processed_data \
    --spacy_model en_core_web_sm \
    --max_batch_size 64 \
    --num_workers 4 \
    --calc_perplexity \
    --perplexity_model meta-llama/Llama-3.2-3B \
    --mixed_precision

