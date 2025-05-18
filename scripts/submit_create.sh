#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --output=logs/create_data_%j.log
#SBATCH --error=logs/create_data_error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# Load environment
module load cray/23.12
module load gcc-native/12.3
module load cray-python/3.11.5
module load cudatoolkit/24.3_12.3

# Set environment variables
# export HUGGING_FACE_HUB_TOKEN=<your-token-here>
export TOKENIZERS_PARALLELISM=false

# Create log directory
mkdir -p logs

# Define dataset paths
DATASETS=(
    "processed_data/cs_bulletin_pdf_en_1014_Llama-3.1-8B_fp16_137666_perplexity_scores.csv"
    "processed_data/cs_bulletin_ocr_en_1009_Llama-3.1-8B_fp16_423396_perplexity_scores.csv"
)

# Define context window range
CONTEXT_WINDOWS=(5 10 20)

# Define base output directory
BASE_OUTPUT_DIR="datasets/word_datasets"

# Loop through different context windows
for window in "${CONTEXT_WINDOWS[@]}"; do
    # Create separate output directory for each context window
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/window_${window}"
    echo "Processing context window = ${window}"

    # Run data processing script
    srun python ../create_data.py \
        --data_paths "${DATASETS[@]}" \
        --model_name "meta-llama/Llama-2-7b-chat-hf" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 32 \
        --max_length 2048 \
        --context_window $window

    # Check run status
    if [ $? -ne 0 ]; then
        echo "Failed while processing context window = ${window}"
        exit 1
    fi
done

echo "All context windows processed"

# Statistics for generated datasets
echo "Generated dataset statistics:"
for window in "${CONTEXT_WINDOWS[@]}"; do
    echo "Context Window = ${window}:"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/window_${window}"
    for dir in "$OUTPUT_DIR"/*; do
        if [ -d "$dir" ]; then
            word_pos=$(basename "$dir")
            num_files=$(find "$dir" -type f | wc -l)
            echo "  $word_pos: $num_files files"
        fi
    done
done 