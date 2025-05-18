#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=stacked_layers_plot
#SBATCH --nodes=1
#SBATCH --output=logs/stacked_layers_plot_%j.log  
#SBATCH --error=logs/stacked_layers_plot_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=<your-account-id>
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00

# Source the config loader and load variables
source ../scripts/load_config.sh
load_yaml_config ../config.yaml

# Load necessary modules
module load python gcc cuda

# Activate virtual environment (Using path from config)
# load your env

# Set environment variables (Using paths/settings from config)
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Define target words (format: word:POS) and list of extraction methods (Using lists from config)
words=($WORDS_GROUP1)
methods=($METHODS_ALL)

# Base directory (adjust according to actual situation) (Using paths from config)
# BASE_HS_DIR loaded from config
# BASE_LABEL_DIR loaded from config
OUTPUT_DIR_BASE="stacked_layers_plots" # Keep specific output base or use $OUTPUT_DIR_BASE
DIM_REDUCE_METHOD=${DEFAULT_DIM_REDUCE_METHOD:-umap} # Use variable from config

# Create log directory (Using path from config)
mkdir -p "$LOG_DIR"

# Iterate through each word and extraction method
for word in "${words[@]}"; do
    # Replace colon with underscore to form directory name (e.g., import:NOUN -> import_NOUN)
    word_folder=$(echo "$word" | sed 's/:/_/g')

    for method in "${methods[@]}"; do
        echo "Starting processing $word (Directory: $word_folder) Extraction method: $method"

        # Hidden states file: Find the hidden states file in the latest hidden_states_* directory under BASE_HS_DIR/word_folder/method/combined
        hs_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/hidden_states_*"
        hs_dir=$(ls -d $hs_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$hs_dir" ]; then
            echo "Hidden states directory matching pattern not found: $hs_dir_pattern"
            continue
        fi
        HIDDEN_STATES_FILE="${hs_dir}/hidden_states_${word_folder}_hidden_states_${method}.pt"
        if [ ! -f "$HIDDEN_STATES_FILE" ]; then
            echo "Hidden states file does not exist: $HIDDEN_STATES_FILE"
            continue
        fi

        # Sentence CSV file: Find the latest icl_basic_${method}_${word_folder}_* directory under BASE_HS_DIR/word_folder/method/combined
        sentence_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/icl_basic_${method}_${word_folder}_*"
        sentence_dir=$(ls -d $sentence_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$sentence_dir" ]; then
            echo "Sentence CSV directory matching pattern not found: $sentence_dir_pattern"
            continue
        fi
        SENTENCE_CSV="${sentence_dir}/icl_basic_${method}_${word_folder}.csv"
        if [ ! -f "$SENTENCE_CSV" ]; then
            echo "Sentence CSV file does not exist: $SENTENCE_CSV"
            continue
        fi

        # Label CSV file: Find the corresponding label file under BASE_LABEL_DIR
        LABEL_CSV="${BASE_LABEL_DIR}/${word_folder}_labeled.csv"
        if [ ! -f "$LABEL_CSV" ]; then
            echo "Label CSV file does not exist: $LABEL_CSV"
            continue
        fi

        # Output directory: Save by word and method
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}_${DIM_REDUCE_METHOD}"
        mkdir -p "$OUTPUT_DIR"

        echo "Generating stacked layer plots for $word ($method)..."
        echo "Hidden states file: $HIDDEN_STATES_FILE"
        echo "Sentence CSV file: $SENTENCE_CSV"
        echo "Label CSV file: $LABEL_CSV"
        echo "Plot output directory: $OUTPUT_DIR"

        # Use srun to call Python script (Using dim method from config)
        srun python ../plot/run_plot_stacked_layers.py \
          --hidden_states_dir "$HIDDEN_STATES_FILE" \
          --sentence_csv "$SENTENCE_CSV" \
          --label_csv "$LABEL_CSV" \
          --output_dir "$OUTPUT_DIR" \
          --method "$DIM_REDUCE_METHOD"

        echo "Processing $word ($method) completed."
    done
done

echo "All tasks completed." 