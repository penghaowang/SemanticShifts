#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_rpcl
#SBATCH --nodes=1
#SBATCH --output=logs/submit_rpcl_%j.log  # Use $LOG_DIR
#SBATCH --error=logs/submit_rpcl_error_%j.log # Use $LOG_DIR    
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

# Define all extraction methods to be processed (Using list from config)
methods=($METHODS_SUBSET1) 

# List of layer configurations to combine (Using list from config - Bash reads the space-separated quoted strings)
layer_configs=($LAYER_CONFIGS_RPCL)

# Dimensionality reduction method (Using list from config)
dimension_methods=($DIMENSION_METHODS_ALL)

# Base directory (adjust according to actual situation) (Using paths from config)
# BASE_LABEL_DIR loaded
# BASE_HS_DIR loaded
OUTPUT_DIR_BASE="combined_layers_plots" # Keep specific base or use $OUTPUT_DIR_BASE

# Create output directory (Using path from config)
mkdir -p "$OUTPUT_DIR_BASE"
mkdir -p "$LOG_DIR"

# Iterate through each word and extraction method
for word in ${words[@]}; do
    # Replace colon with underscore to form directory name (e.g., import:NOUN -> import_NOUN)
    word_folder=$(echo "$word" | sed 's/:/_/g')
    
    # Note: The original script had 'method' uninitialized here, likely a bug.
    # Assuming loop over methods is intended.
    for method in ${methods[@]}; do
        echo "Starting processing $word (Directory: $word_folder) Extraction method: $method"
        # Hidden states directory: Find the latest hidden_states_* directory under BASE_HS_DIR/word_folder/method/combined
        hs_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/hidden_states_*"
        hs_dir=$(ls -d $hs_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$hs_dir" ]; then
            echo "Hidden states directory matching pattern not found: $hs_dir_pattern"
            continue
        fi
        HIDDEN_STATES_DIR="${hs_dir}"
        if [ ! -d "$HIDDEN_STATES_DIR" ]; then
            echo "Hidden states directory does not exist: $HIDDEN_STATES_DIR"
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

        # Process for each layer configuration and dimensionality reduction method
        for layers_config in "${layer_configs[@]}"; do # Iterate over array from config
            # Remove potential outer quotes added by bash for space-separated strings
            layers=$(echo $layers_config | sed "s/^'//;s/'$//")
            for dimension_method in ${dimension_methods[@]}; do # Iterate over space-separated string
                echo "Processing $word ($method) - Layer Config: $layers - Dim Reduction: $dimension_method"
                
                # Replace commas and hyphens in layer config with underscores for filename
                layer_name=$(echo "$layers" | sed 's/,/_/g' | sed 's/-/to/g')
                
                # Output directory, includes layer configuration information
                OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}_${dimension_method}_layers_${layer_name}"
                mkdir -p "$OUTPUT_DIR"
                
                # Use srun to call Python script
                srun python ../plot/run_plot_comb_layers.py \
                  --hidden_states_dir "$HIDDEN_STATES_DIR" \
                  --sentence_csv "$SENTENCE_CSV" \
                  --label_csv "$LABEL_CSV" \
                  --output_dir "$OUTPUT_DIR" \
                  --layers "$layers" \
                  --method "$dimension_method" \
                  --word "$word" \
                  --extraction_method "$method"
                
                echo "Completed $word ($method) - Layer $layers - Dim Reduction $dimension_method"
            done
        done
    done
done

echo "All tasks completed." 