#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_plot
#SBATCH --nodes=1
#SBATCH --output=logs/submit_plot_%j.log  
#SBATCH --error=logs/submit_plot_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=<your-account-id>
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=12:00:00

# Load necessary modules
module load python gcc cuda

# load your env
# source /path/to/your/venv/bin/activate

# Set environment variables
# export HUGGING_FACE_HUB_TOKEN=<your-token-here>
# export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/path/to/your/hf/cache"

# Define target words (format: word:POS) and list of extraction methods
# words=("import:NOUN" "export:NOUN" "money:NOUN" "price:NOUN" "product:NOUN" 
#     "sale:NOUN" "agreement:NOUN" "annual:ADJ" "financial:ADJ" "net:ADJ" 
#     "industrial:ADJ" "traditional:ADJ" "monetary:ADJ" "inflationary:ADJ" "foreign:ADJ" 
#     "public:ADJ" "private:ADJ" "corporate:ADJ" "real:ADJ" "available:ADJ" 
#     "strong:ADJ" "stable:ADJ" "fair:ADJ" "competitive:ADJ")
words=("market:NOUN" "rate:NOUN" "bank:NOUN" "interest:NOUN" "investment:NOUN"
    "bond:NOUN" "share:NOUN" "capital:NOUN" "exchange:NOUN" "tax:NOUN"
    "growth:NOUN" "security:NOUN" "company:NOUN" "dollar:NOUN" "debt:NOUN"
    "equity:NOUN" "profit:NOUN" "loss:NOUN" "gain:NOUN" "decline:NOUN")
methods=("input_last_token" "eos_token" "input_mean" "output_mean" "output_eos")

# Base directory (adjust according to actual situation)
BASE_HS_DIR="hidden_states"
BASE_LABEL_DIR="labeled_data"
OUTPUT_DIR_BASE="api_hs_plots"
DIM_REDUCE_METHOD="umap"

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

        # Label CSV file: Assume it is under BASE_LABEL_DIR, with filename format ${word_folder}_labeled.csv
        LABEL_CSV="${BASE_LABEL_DIR}/${word_folder}_labeled.csv"
        if [ ! -f "$LABEL_CSV" ]; then
            echo "Label CSV file does not exist: $LABEL_CSV"
            continue
        fi

        # Output directory: Save by word and method
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}_${DIM_REDUCE_METHOD}"
        mkdir -p "$OUTPUT_DIR"

        echo "Generating plots for $word ($method)..."
        echo "Hidden states file: $HIDDEN_STATES_FILE"
        echo "Sentence CSV file: $SENTENCE_CSV"
        echo "Label CSV file: $LABEL_CSV"
        echo "Plot output directory: $OUTPUT_DIR"

        # Use srun to call Python script
        srun python ../plot/run_plot.py \
          --hidden_states_dir "$HIDDEN_STATES_FILE" \
          --sentence_csv "$SENTENCE_CSV" \
          --label_csv "$LABEL_CSV" \
          --output_dir "$OUTPUT_DIR" \
          --method "$DIM_REDUCE_METHOD"

        echo "Processing $word ($method) completed."
    done
done

echo "All tasks completed."
