#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=all_words_traj
#SBATCH --nodes=1
#SBATCH --output=logs/all_words_traj_%j.log  
#SBATCH --error=logs/all_words_traj_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=<your-account-id>
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00
# First, integrate high-dimensional vectors from all time points and perform unified dimensionality reduction, then extract coordinates to plot the trajectory. This method fully utilizes UMAP's global structure preservation ability, ensuring the spatio-temporal continuity of semantic changes is accurately represented in the low-dimensional space.
# Load necessary modules
module load python gcc cuda

# Activate virtual environment
# load your env

# Set environment variables
# export HUGGING_FACE_HUB_TOKEN=<your-token-here>
export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/path/to/your/hf/cache"

# Base directory
BASE_HS_DIR="hidden_states"
OUTPUT_DIR_BASE="all_words_trajectory_plots"

# Dimensionality reduction method
dimension_method="umap"

# Extraction method
extraction_methods=("input_last_token" "eos_token")

# Arrow size
arrow_scale=0.15

# Core words in the financial domain
word_list=(
    "market" "rate" "bank" "interest" "investment"
    "bond" "share" "capital" "exchange" "tax"
    "growth" "security" "company" "dollar" "debt"
    "equity" "profit" "loss" "gain" "decline"
    "import" "export" "money" "price" "product"
    "sale" "agreement" "annual" "financial" "net"
    "industrial" "traditional" "monetary" "inflationary" "foreign"
    "public" "private" "corporate" "real" "available"
    "strong" "stable" "fair" "competitive"
)

# Create output directory
mkdir -p "$OUTPUT_DIR_BASE"
mkdir -p logs

# Check if Python script exists
if [ ! -f "../plot/plot_word_trajectory.py" ]; then
    echo "Error: plot_word_trajectory.py file does not exist"
    exit 1
fi

# Run once for each extraction method
for extraction_method in "${extraction_methods[@]}"; do
    echo "Starting processing - Extraction method: $extraction_method - Dim Reduction: $dimension_method"

    # Output directory, includes parameter information
    output_dir="${OUTPUT_DIR_BASE}/${extraction_method}_${dimension_method}"
    mkdir -p "$output_dir"

    # Build word list arguments
    word_list_args=""
    for word in "${word_list[@]}"; do
        word_list_args+="$word "
    done

    # Use srun to run Python script
    srun python ../plot/plot_word_trajectory.py \
        --base_dir "$BASE_HS_DIR" \
        --output_dir "$output_dir" \
        --method "$dimension_method" \
        --extraction_method "$extraction_method" \
        --arrow_scale "$arrow_scale" \
        --word_list $word_list_args

    # Check the exit status of the previous command
    if [ $? -ne 0 ]; then
        echo "Error: An error occurred while processing $extraction_method"
        continue
    fi

    echo "Completed - Extraction method: $extraction_method - Dim Reduction: $dimension_method"
done

echo "All vocabulary trajectory plot drawing tasks completed." 