#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_layer_plot
#SBATCH --nodes=1
#SBATCH --output=logs/submit_layer_plot_%j.log  
#SBATCH --error=logs/submit_layer_plot_error_%j.log     
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


# Define list of extraction methods (Using list from config)
methods=($METHODS_ALL)

# Base directory (adjust according to actual situation) (Using paths from config)
# BASE_HS_DIR loaded
OUTPUT_DIR_BASE="layer_plots" # Keep specific base or use $OUTPUT_DIR_BASE

# Create log directory (Using path from config)
mkdir -p "$LOG_DIR"

# Iterate through each extraction method
for method in ${methods[@]}; do
    echo "Starting processing extraction method: $method"

    # Create output directory
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${method}"
    mkdir -p "$OUTPUT_DIR"

    echo "Performing inter-layer similarity analysis for extraction method $method..."

    # Run inter-layer similarity analysis script
    python ../plot/run_layer_plot.py \
        --base_hs_dir "$BASE_HS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --method "$method"

    echo "Completed inter-layer similarity analysis for extraction method $method, results saved in $OUTPUT_DIR"
done

echo "Inter-layer similarity analysis for all extraction methods completed"
