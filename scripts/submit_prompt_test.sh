#!/bin/bash

#SBATCH --job-name=prompt_test
#SBATCH --output=logs/test_prompt_%j.log
#SBATCH --error=logs/test_prompt_error_%j.log
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1              
#SBATCH --time=12:00:00                 
#SBATCH --partition=normal   
#SBATCH --constraint=gpu

# Set test parameters
WORD_TYPE="gain:NOUN"
NUM_SAMPLES=20
OUTPUT_DIR="outputs/prompt_test"
EXTRACTION_METHOD="input_last_token"

# Iterate through different prompt templates
for PROMPT_TEMPLATE in "basic" "context_aware" "finance" "icl_basic" "icl_detailed"
do
    echo "Testing prompt template: ${PROMPT_TEMPLATE}"
    
    python ../run_inference.py \
        --word_type ${WORD_TYPE} \
        --num_samples ${NUM_SAMPLES} \
        --prompt_template ${PROMPT_TEMPLATE} \
        --output_dir "${OUTPUT_DIR}/${PROMPT_TEMPLATE}" \
        --extraction_method ${EXTRACTION_METHOD} \
        --save_hidden_states
done

echo "All prompt template tests completed!" 