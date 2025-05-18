#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_wordtraj
#SBATCH --nodes=1
#SBATCH --output=logs/submit_wordtraj_%j.log  
#SBATCH --error=logs/submit_wordtraj_error_%j.log     
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
words=($WORDS_GROUP1) # Use group1 words from config

# Define all extraction methods to be processed (Using list from config)
methods=($METHODS_SUBSET1) # Use subset from config

# Dimensionality reduction method (Using list from config)
dimension_methods=($DIMENSION_METHODS_ALL)

# Time binning level (Using list from config)
time_bin_levels=($TIME_BIN_LEVELS)

# Arrow size (Using value from config)
arrow_scale=${DEFAULT_ARROW_SCALE:-0.15}

# Base directory (adjust according to actual situation) (Using paths from config)
# BASE_LABEL_DIR loaded from config
# BASE_HS_DIR loaded from config
OUTPUT_DIR_BASE="word_trajectory_plots" # Keep specific output base or use $OUTPUT_DIR_BASE

# Create output directory (Using path from config)
mkdir -p "$OUTPUT_DIR_BASE"
mkdir -p "$LOG_DIR"

# Iterate through each word and extraction method
for word in ${words[@]}; do # Iterate over space-separated string
    # Extract word and POS tag
    word_part=$(echo "$word" | cut -d':' -f1)
    pos_part=$(echo "$word" | cut -d':' -f2)
    
    # Replace colon with underscore to form directory name (e.g., import:NOUN -> import_NOUN)
    word_folder=$(echo "$word" | sed 's/:/_/g')
    
    echo "Starting processing $word (Directory: $word_folder)"
    
    for method in ${methods[@]}; do # Iterate over space-separated string
        echo "Using extraction method: $method"
        
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

        # Process for each dimensionality reduction method and time binning level
        for dim_method in ${dimension_methods[@]}; do # Iterate over space-separated string
            for time_bin in ${time_bin_levels[@]}; do # Iterate over space-separated string
                echo "Processing $word ($method) - Dim Reduction: $dim_method - Time Bin: $time_bin"
                
                # Output directory, includes all parameter information
                OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}_${dim_method}_${time_bin}"
                mkdir -p "$OUTPUT_DIR"
                
                # Use srun to call Python script (Using arrow_scale from config)
                srun python ../plot/plot_word_trajectory.py \
                  --hidden_states_dir "$HIDDEN_STATES_DIR" \
                  --sentence_csv "$SENTENCE_CSV" \
                  --label_csv "$LABEL_CSV" \
                  --output_dir "$OUTPUT_DIR" \
                  --method "$dim_method" \
                  --word "$word_part" \
                  --pos "$pos_part" \
                  --arrow_scale "$arrow_scale" \
                  --time_bin_level "$time_bin"
                
                echo "Completed $word ($method) - Dim Reduction $dim_method - Time Bin $time_bin"
            done
        done
    done
done

echo "Trajectory plot generation task for all words completed." 