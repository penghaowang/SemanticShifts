#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_rpcl
#SBATCH --nodes=1
#SBATCH --output=logs/submit_rpcl_%j.log  
#SBATCH --error=logs/submit_rpcl_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=<your-account-id>
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00

# Load necessary modules
module load python gcc cuda

# load your env
# source /path/to/your/venv/bin/activate

# REMOVED: Set env vars (e.g., HUGGING_FACE_HUB_TOKEN, HF_CACHE_PATH, BASE_STORAGE_PATH) outside the script
# export HUGGING_FACE_HUB_TOKEN=<your-token-here> # Removed hardcoded token
# export TOKENIZERS_PARALLELISM=false
# export HF_HOME="${HF_CACHE_PATH:-/path/to/your/hf/cache}" # Use HF_CACHE_PATH env var, fallback to original if not set

# REMOVED: Parameters defined in config YAML
# Set basic parameters
# BASE_HS_DIR="${BASE_STORAGE_PATH:-/path/to/your/hidden_states}" # Use BASE_STORAGE_PATH env var
OUTPUT_DIR="method_correlations" # Keep base output dir for script output
# BATCH_SIZE=100
# FILE_PATTERN="*.pt"
# Define list of extraction methods
# METHODS=("input_last_token" "eos_token" "input_mean" "output_mean" "output_eos")
# Define layer indices to analyze (0-based)
# LAYER_INDICES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"

# REMOVED: Logic for finding word folders moved to Python script based on config
# echo "Searching for all vocabulary directories..."
# WORD_FOLDERS=()
# for dir in "$BASE_HS_DIR"/*; do
#   if [ -d "$dir" ]; then
#     word_folder=$(basename "$dir")
#     for method in "${METHODS[@]}"; do
#       if [ -d "$dir/$method" ]; then
#         WORD_FOLDERS+=("$word_folder")
#         break
#       fi
#     done
#   fi
# done
# if [ ${#WORD_FOLDERS[@]} -eq 0 ]; then
#   echo "Error: No valid vocabulary directories found"
#   exit 1
# fi

# Print parameter information - REMOVED: Parameters are in config
# echo "======================= Extraction Method Correlation Analysis ======================="
# echo "Base hidden state directory: $BASE_HS_DIR"
# echo "List of extraction methods: ${METHODS[*]}"
# echo "Specified layer indices: $LAYER_INDICES"
# echo "Found ${#WORD_FOLDERS[@]} vocabulary directories"
# echo "Vocabulary list: ${WORD_FOLDERS[*]}"
# echo "Batch size: $BATCH_SIZE"
# echo "File matching pattern: $FILE_PATTERN"
# echo "Output directory: $OUTPUT_DIR"
# echo "==============================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# REMOVED: Layer param construction based on removed variable
# LAYER_PARAM=""
# if [ -n "$LAYER_INDICES" ]; then
#   LAYER_PARAM="--layer_indices $LAYER_INDICES"
# fi

# Run Python script
python ../runners/plotting/run_plot_correlation.py \
  --config "config/correlation_config.yaml" \
  --output_dir "$OUTPUT_DIR"
  # --base_hs_dir "$BASE_HS_DIR" \ # Removed, from config
  # --methods "${METHODS[@]}" \ # Removed, from config
  # --word_folders "${WORD_FOLDERS[@]}" \ # Removed, Python script finds these
  # $LAYER_PARAM \ # Removed, from config
  # --batch_size $BATCH_SIZE \ # Removed, from config
  # --file_pattern "$FILE_PATTERN" # Removed, from config

# Check execution result
if [ $? -eq 0 ]; then
  echo "Correlation analysis completed successfully!"
  echo "Results for each word are saved in the $OUTPUT_DIR/[word_name]/ directory"
  echo "Combined results for all words are saved in the $OUTPUT_DIR/ directory"
  
  # Check if the overall heatmap was generated
  if [ -f "$OUTPUT_DIR/all_words_method_correlation_heatmap.png" ]; then
    echo "Overall heatmap generated: $OUTPUT_DIR/all_words_method_correlation_heatmap.png"
  else
    echo "Warning: Overall heatmap was not generated, possibly due to insufficient valid data"
  fi
else
  echo "Correlation analysis failed, please check the log file for details."
fi 