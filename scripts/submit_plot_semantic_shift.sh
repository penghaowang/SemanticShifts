#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=semantic_shift
#SBATCH --nodes=1
#SBATCH --output=logs/semantic_shift_%j.log  
#SBATCH --error=logs/semantic_shift_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=<your-account-id>
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=01:00:00
# This script is used for batch processing semantic shift analysis of vocabulary, plotting JS divergence and silhouette coefficient changes

# Load necessary modules
module load python gcc cuda

# Activate virtual environment
# load your env

# Set environment variables
# export HUGGING_FACE_HUB_TOKEN=<your-token-here>
export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/path/to/your/hf/cache"

# Basic parameters
BASE_DIR="hidden_states"
OUTPUT_BASE_DIR="semantic_shift_plots"
METHOD="umap"
LAYER_IDX=-1
TIME_BIN_LEVEL="period"  # Optional "year" or "period"
EXTRACTION_METHOD="input_last_token"  # Keep consistent with plot_word_trajectory.py

# Core words and their POS tags in the financial domain
declare -A word_pos
# Use vocabulary list from file_context_0
for word in "market" "rate" "bank" "interest" "investment" "bond" "share" "capital" "exchange" "tax" "growth" "security" "company" "dollar" "debt" "equity" "gain" "decline" "import" "export" "money" "price" "product" "sale" "agreement" "annual" "financial" "net" "industrial" "traditional" "monetary" "inflationary" "foreign" "public" "private" "corporate" "real" "available" "strong" "stable" "fair" "competitive" "profit" "loss"; do
    # Default setting is noun, specific adjectives are handled separately
    word_pos[$word]="NOUN"

    # Adjective list processing
    if [[ "$word" == "financial" || "$word" == "net" || "$word" == "industrial" || "$word" == "traditional" || "$word" == "monetary" || "$word" == "inflationary" || "$word" == "foreign" || "$word" == "public" || "$word" == "private" || "$word" == "corporate" || "$word" == "real" || "$word" == "available" || "$word" == "strong" || "$word" == "stable" || "$word" == "fair" || "$word" == "competitive" || "$word" == "annual" ]]; then
        word_pos[$word]="ADJ"
    fi
done

# Create output directory and log directory
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p logs

# Check if Python script exists
if [ ! -f "shift_plot.py" ]; then
    echo "Error: shift_plot.py file does not exist"
    exit 1
fi

echo "Starting vocabulary semantic shift analysis - Method: $METHOD - Time Grouping: $TIME_BIN_LEVEL"

# Process each vocabulary word
for word in "${!word_pos[@]}"; do
    pos=${word_pos[$word]}
    echo "Processing word: $word (POS: $pos)"

    # Remove quotes from the word (if any)
    clean_word=$(echo $word | tr -d '\"'')

    # Create output directory for each word
    word_output_dir="${OUTPUT_BASE_DIR}/${clean_word}"
    mkdir -p "$word_output_dir"

    # Construct hidden state directory path for the word
    WORD_DIR="${BASE_DIR}/${clean_word}_${pos}"

    # Hidden state directory
    HS_DIR_PATTERN="${WORD_DIR}/${EXTRACTION_METHOD}/combined/hidden_states_*"
    HS_DIRS=$(ls -d $HS_DIR_PATTERN 2>/dev/null)

    if [ -z "$HS_DIRS" ]; then
        echo "Warning: Hidden state directory not found: $HS_DIR_PATTERN"
        continue
    fi

    # Use the latest hidden state directory
    HS_DIR=$(echo $HS_DIRS | tr ' ' '\\n' | sort | tail -n 1)

    # Find hidden state files
    PT_FILES=$(ls ${HS_DIR}/*.pt 2>/dev/null)
    if [ -z "$PT_FILES" ]; then
        echo "Warning: No .pt file found in ${HS_DIR}"
        continue
    fi

    # Use the first .pt file
    HS_FILE=$(echo $PT_FILES | tr ' ' '\\n' | head -n 1)

    # Sentence CSV file directory
    SENTENCE_DIR_PATTERN="${WORD_DIR}/${EXTRACTION_METHOD}/combined/icl_basic_${EXTRACTION_METHOD}_${clean_word}_${pos}_*"
    SENTENCE_DIRS=$(ls -d $SENTENCE_DIR_PATTERN 2>/dev/null)

    if [ -z "$SENTENCE_DIRS" ]; then
        echo "Warning: Sentence CSV directory not found: $SENTENCE_DIR_PATTERN"
        continue
    fi

    # Use the latest sentence directory
    SENTENCE_DIR=$(echo $SENTENCE_DIRS | tr ' ' '\\n' | sort | tail -n 1)
    SENTENCE_CSV="${SENTENCE_DIR}/icl_basic_${EXTRACTION_METHOD}_${clean_word}_${pos}.csv"

    # Label CSV file
    LABEL_CSV="labeled_data/${clean_word}_${pos}_labeled.csv"

    if [ ! -f "$SENTENCE_CSV" ]; then
        echo "Warning: Sentence CSV file does not exist: $SENTENCE_CSV"
        continue
    fi

    if [ ! -f "$LABEL_CSV" ]; then
        echo "Warning: Label CSV file does not exist: $LABEL_CSV"
        continue
    fi

    # Run Python script
    srun python ../plot/run_shift_plot.py \
        --hidden_states_dir "$HS_DIR" \
        --sentence_csv "$SENTENCE_CSV" \
        --label_csv "$LABEL_CSV" \
        --output_dir "$word_output_dir" \
        --method "$METHOD" \
        --layer_idx "$LAYER_IDX" \
        --word "$clean_word" \
        --time_bin_level "$TIME_BIN_LEVEL"

    # Check the exit status of the previous command
    if [ $? -ne 0 ]; then
        echo "Error: An error occurred while processing word $clean_word"
        continue
    fi

    echo "Completed word: $clean_word"
done

echo "Semantic shift analysis tasks for all words completed." 

# Collect and summarize JSD and entropy data for all words
echo "Starting to collect and summarize semantic shift data for all words..."

# Check if collect_semantic_data.py exists
if [ ! -f "collect_semantic_data.py" ]; then
    echo "Error: collect_semantic_data.py file does not exist, cannot collect summary data"
    exit 1
fi

# Run data collection script
srun python ../collect_semantic_data.py

# Check script execution status
if [ $? -eq 0 ]; then
    echo "Data collection and summarization completed successfully. Summary data saved in the semantic_shift_summary directory."
else
    echo "Error: A problem occurred during data collection and summarization"
    exit 1
fi

echo "The entire analysis process is complete." 