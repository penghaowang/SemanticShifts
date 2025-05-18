#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_rpd
#SBATCH --nodes=1
#SBATCH --output=logs/submit_rpd_%j.log  
#SBATCH --error=logs/submit_rpd_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=<your-account-id>
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00

# Load necessary modules
module load python gcc cuda

# Activate virtual environment
# load your env

# Set environment variables
# export HUGGING_FACE_HUB_TOKEN=<your-token-here>
export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/path/to/your/hf/cache"

# Define target words (format: word:POS) and list of extraction methods
words=(
    # First group of words - Nouns
    "market:NOUN" "rate:NOUN" "bank:NOUN" "interest:NOUN" "investment:NOUN"
    "bond:NOUN" "share:NOUN" "capital:NOUN" "exchange:NOUN" "tax:NOUN"
    "growth:NOUN" "security:NOUN" "company:NOUN" "dollar:NOUN" "debt:NOUN"
    "equity:NOUN" "profit:NOUN" "loss:NOUN" "gain:NOUN" "decline:NOUN"
    # Second group of words - Nouns and Adjectives
    "import:NOUN" "export:NOUN" "money:NOUN" "price:NOUN" "product:NOUN" 
    "sale:NOUN" "agreement:NOUN" "annual:ADJ" "financial:ADJ" "net:ADJ" 
    "industrial:ADJ" "traditional:ADJ" "monetary:ADJ" "inflationary:ADJ" "foreign:ADJ" 
    "public:ADJ" "private:ADJ" "corporate:ADJ" "real:ADJ" "available:ADJ" 
    "strong:ADJ" "stable:ADJ" "fair:ADJ" "competitive:ADJ"
)
methods=("input_last_token" "eos_token" "input_mean" "output_mean" "output_eos")

# Base directory (adjust according to actual situation)
BASE_LABEL_DIR="labeled_data"
BASE_HS_DIR="hidden_states"
OUTPUT_DIR_BASE="diachronic_plots"
DIM_REDUCE_METHOD="umap"  # Optional: 'pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'

# Define layer configuration
LAYER_CONFIGS=(
    "24-31"
)

# Create log directory
mkdir -p logs

# Iterate through each word and extraction method
for word in "${words[@]}"; do
    # Replace colon with underscore to form directory name (e.g., import:NOUN -> import_NOUN)
    word_folder=$(echo "$word" | sed 's/:/_/g')

    echo "Starting processing $word (Directory: $word_folder) Extraction method: $method"

    for method in "${methods[@]}"; do
        for layer_config in "${LAYER_CONFIGS[@]}"; do
            echo "Processing $word ($method) - Layer Config: $layer_config - Dim Reduction: $DIM_REDUCE_METHOD"

            # Hidden states directory: Find the latest hidden_states_* directory under BASE_HS_DIR/word_folder/method/combined
            hs_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/hidden_states_*"
            hs_dir=$(ls -d $hs_dir_pattern 2>/dev/null | sort | tail -n 1)
            if [ -z "$hs_dir" ]; then
                echo "Hidden states directory matching pattern not found: $hs_dir_pattern"
                continue
            fi
            HIDDEN_STATES_FILE="${hs_dir}"
            if [ ! -d "$HIDDEN_STATES_FILE" ]; then
                echo "Hidden states directory does not exist: $HIDDEN_STATES_FILE"
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

            # Output directory: Save by word, method, and layer configuration
            OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}/${layer_config}_${DIM_REDUCE_METHOD}"
            mkdir -p "$OUTPUT_DIR"

            # Use srun to call Python script
            srun python ../plot/run_plot_diachronic.py \
              --hidden_states_dir "$HIDDEN_STATES_FILE" \
              --sentence_csv "$SENTENCE_CSV" \
              --label_csv "$LABEL_CSV" \
              --output_dir "$OUTPUT_DIR" \
              --method "$DIM_REDUCE_METHOD" \
              --layers "$layer_config" \
              --word "$word" \
              --extraction_method "$method"

            echo "Completed $word ($method) - Layer $layer_config - Dim Reduction $DIM_REDUCE_METHOD"
        done
        echo "Processing $word ($method) completed."
    done
done

echo "All tasks completed."