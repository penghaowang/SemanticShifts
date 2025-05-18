#!/bin/bash
#SBATCH --job-name=word_trajectory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/word_trajectory_%j.out
#SBATCH --error=logs/word_trajectory_%j.err
#SBATCH --partition=normal

# Load necessary modules
module load python/3.9
module load cuda/11.7

# Get command line arguments
HIDDEN_STATES_DIR=$1  # Hidden states directory
SENTENCE_CSV=$2       # Sentence CSV file
LABEL_CSV=$3          # Label CSV file
OUTPUT_DIR=$4         # Output directory
WORD=${5:-""}         # Target word (optional)
POS=${6:-""}          # POS tag (optional)
METHOD=${7:-"umap"}   # Dimensionality reduction method (default umap)
LAYER_IDX=${8:--1}    # Layer index (default -1, means average over all layers)
TIME_BIN=${9:-"year"} # Time binning level (default by year)
ARROW_SCALE=${10:-0.1} # Arrow scale (default 0.1)

# Ensure log directory exists
mkdir -p logs

# Print parameters
echo "----------- Task Parameters ------------"
echo "Hidden States Directory: $HIDDEN_STATES_DIR"
echo "Sentence CSV File: $SENTENCE_CSV"
echo "Label CSV File: $LABEL_CSV"
echo "Output Directory: $OUTPUT_DIR"
echo "Target Word: $WORD"
echo "POS Tag: $POS"
echo "Reduction Method: $METHOD"
echo "Layer Index: $LAYER_IDX"
echo "Time Binning Level: $TIME_BIN"
echo "Arrow Scale: $ARROW_SCALE"
echo "---------------------------------"

# Build arguments string
ARGS="--mode single --hidden_states_dir $HIDDEN_STATES_DIR --sentence_csv $SENTENCE_CSV --label_csv $LABEL_CSV --output_dir $OUTPUT_DIR --method $METHOD --layer_idx $LAYER_IDX --time_bin_level $TIME_BIN --arrow_scale $ARROW_SCALE"

# Add optional arguments
if [ ! -z "$WORD" ]; then
  ARGS="$ARGS --word $WORD"
fi

if [ ! -z "$POS" ]; then
  ARGS="$ARGS --pos $POS"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run Python script
echo "Starting to plot word trajectory..."
python ../plot/plot_word_trajectory.py $ARGS

echo "Task completed!"

# Usage example:
# sbatch submit_word_trajectory.sh /path/to/hidden_states path/to/sentences.csv path/to/labels.csv output_dir bank NOUN umap -1 period 0.15 