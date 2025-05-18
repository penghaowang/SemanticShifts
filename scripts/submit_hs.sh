#!/bin/bash

#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_hs
#SBATCH --nodes=1
#SBATCH --output=logs/submit_hs_%j.log  
#SBATCH --error=logs/submit_hs_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=<your-account-id>
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=01:30:00

module load python gcc cuda
# load your env

# Set environment variables
# export HUGGING_FACE_HUB_TOKEN=<your-token-here>
export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/path/to/your/hf/cache"

# Set log directory
LOG_DIR="logs/hidden_states"
mkdir -p $LOG_DIR

# Test configuration
DATA_PATHS=("data/combined.csv")
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Basic parameters
BATCH_SIZE=8
TEMPERATURE=0.3
MAX_TOKENS=50
PROMPT_TEMPLATE="icl_basic"  # Add as hyperparameter

# Different context window sizes
CONTEXT_WINDOWS=(0 1 3 5 10)

# Modify layer index format, use comma separation instead of space
LAYER_INDICES="30"

# Target word list
# TARGET_WORDS=(
#     "import:NOUN" "export:NOUN" "money:NOUN" "price:NOUN" "product:NOUN"
#     "sale:NOUN" "financial:ADJ" "net:ADJ"
#     "industrial:ADJ" "traditional:ADJ" "monetary:ADJ" "inflationary:ADJ" "foreign:ADJ"
#     "public:ADJ" "private:ADJ" "corporate:ADJ" "real:ADJ" "available:ADJ"
#     "strong:ADJ" "stable:ADJ" "fair:ADJ" "competitive:ADJ"
#     "gain:NOUN" "loss:NOUN"
# )
# words=("import:NOUN" "export:NOUN" "money:NOUN" "price:NOUN" "product:NOUN" 
#     "sale:NOUN" "agreement:NOUN" "annual:ADJ" "financial:ADJ" "net:ADJ" 
#     "industrial:ADJ" "traditional:ADJ" "monetary:ADJ" "inflationary:ADJ" "foreign:ADJ" 
#     "public:ADJ" "private:ADJ" "corporate:ADJ" "real:ADJ" "available:ADJ" 
#     "strong:ADJ" "stable:ADJ" "fair:ADJ" "competitive:ADJ")
words=("market:NOUN" "rate:NOUN" "fair:ADJ")

# List of extraction methods
EXTRACTION_METHODS=("output_mean")

# Define function to run a single task
run_task() {
    local target_word=$1
    local method=$2
    local data_path=$3
    local gpu_id=$4
    local context_window=$5
    
    local data_file=$(basename "$data_path")
    local data_name="${data_file%.*}"  # Remove extension
    
    # Replace colon in target word with underscore
    local safe_target_word=${target_word//:/_}
    
    echo "=== Task Start ===" | tee -a "$LOG_DIR/test.log"
    echo "GPU $gpu_id: ${target_word} - ${method} - ${data_path} - Context window: ${context_window}" | tee -a "$LOG_DIR/test.log"

    OUTPUT_DIR="hidden_states/hidden_states_llama38B_cw${context_window}_${PROMPT_TEMPLATE}/${safe_target_word}/${method}/${data_name}"
    mkdir -p "$OUTPUT_DIR"

    {
        CUDA_VISIBLE_DEVICES=$gpu_id python ../run_inference.py \
            --data_path "$data_path" \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE \
            --num_beams 1 \
            --temperature $TEMPERATURE \
            --max_new_tokens $MAX_TOKENS \
            --layer_indices "$LAYER_INDICES" \
            --prompt_template "$PROMPT_TEMPLATE" \
            --target_words "$target_word" \
            --context_mode "sentence" \
            --context_window $context_window \
            --duplicate_handling "remove" \
            --extraction_method "$method" \
            --save_hidden_states \
            2>&1
    } | tee "$LOG_DIR/${safe_target_word}_${method}_${data_name}_cw${context_window}_gpu${gpu_id}.log"

    # Check run status
    if [ $? -eq 0 ]; then
        echo "✓ Task Completed" | tee -a "$LOG_DIR/test.log"
    else:
        echo "✗ Task Failed" | tee -a "$LOG_DIR/test.log"
    fi

    echo "=============" | tee -a "$LOG_DIR/test.log"
}

# Iterate through target words
for TARGET_WORD in "${words[@]}"
do
    echo "Starting processing: ${TARGET_WORD}" | tee -a "$LOG_DIR/test.log"
    echo "=============" | tee -a "$LOG_DIR/test.log"

    # Use an array to track background processes
    declare -a pids
    gpu_id=0

    # Iterate through context window sizes
    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"
    do
        # Iterate through extraction methods
        for METHOD in "${EXTRACTION_METHODS[@]}"
        do
            # Iterate through data paths
            for DATA_PATH in "${DATA_PATHS[@]}"
            do
                # Run task on specified GPU and put in background
                run_task "$TARGET_WORD" "$METHOD" "$DATA_PATH" "$gpu_id" "$CONTEXT_WINDOW" &
                pids+=($!)

                # Update GPU ID, loop through 0-3
                gpu_id=$(( (gpu_id + 1) % 4 ))

                # If 4 tasks have been started, wait for one to complete
                if [ ${#pids[@]} -eq 4 ]; then
                    wait -n  # Wait for any child process to complete
                    # Remove completed process ID
                    for pid in "${pids[@]}"; do
                        if ! kill -0 $pid 2>/dev/null; then
                            pids=("${pids[@]/$pid}")
                        fi
                    done
                fi
            done
        done
    done

    # Wait for all remaining tasks to complete
    wait
done

echo "All processing complete" | tee -a "$LOG_DIR/test.log" 