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
# source /path/to/your/venv/bin/activate

# Set environment variables - do this in your environment, not in the script
# export HUGGING_FACE_HUB_TOKEN=<your-token-here>
# export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/path/to/your/hf/cache"

# Set log directory
LOG_DIR="logs/hidden_states"
mkdir -p $LOG_DIR

# REMOVED: Parameters defined in config YAML
# Test configuration
# DATA_PATHS=("data/combined.csv")
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# Basic parameters
# BATCH_SIZE=8
# TEMPERATURE=0.3
# MAX_TOKENS=50
# PROMPT_TEMPLATE="icl_basic"  # Add as hyperparameter
# Different context window sizes
# CONTEXT_WINDOWS=(0 1 3 5 10)
# Modify layer index format, use comma separation instead of space
# LAYER_INDICES="30"
# List of extraction methods
# EXTRACTION_METHODS=("output_mean")

# Target word list - Kept here as it drives the main loop, could also be moved to config
words=("market:NOUN" "rate:NOUN" "fair:ADJ")

# Define context windows and methods here, or move to config
CONTEXT_WINDOWS=(0 1 3 5 10)
EXTRACTION_METHODS=("output_mean")

# Define function to run a single task
run_task() {
    local target_word=$1
    local method=$2
    # local data_path=$3 # Removed, assume from config
    local gpu_id=$4
    local context_window=$5
    
    # local data_file=$(basename "$data_path") # Removed, assume from config
    # local data_name="${data_file%.*}"  # Remove extension
    local data_name="combined" # Placeholder, Python script should determine input data based on config

    # Replace colon in target word with underscore
    local safe_target_word=${target_word//:/_}
    
    echo "=== Task Start ===" | tee -a "$LOG_DIR/test.log"
    # echo "GPU $gpu_id: ${target_word} - ${method} - ${data_path} - Context window: ${context_window}" | tee -a "$LOG_DIR/test.log" # Removed data_path
    echo "GPU $gpu_id: ${target_word} - ${method} - Context window: ${context_window}" | tee -a "$LOG_DIR/test.log"

    # REMOVED: Output dir construction moved to Python script based on config
    # OUTPUT_DIR="${BASE_STORAGE_PATH:-/path/to/your/hidden_states}/hidden_states_llama38B_cw${context_window}_${PROMPT_TEMPLATE}/${safe_target_word}/${method}/${data_name}" # Use BASE_STORAGE_PATH env var
    # mkdir -p "$OUTPUT_DIR"

    {
        # Pass only dynamic args and config path
        CUDA_VISIBLE_DEVICES=$gpu_id python ../runners/processing/run_inference.py \
            --config "config/inference_config.yaml" \
            --target_words "$target_word" \
            --extraction_method "$method" \
            --context_window $context_window \
            --gpu_id $gpu_id
            # --data_path "$data_path" \ # Removed, from config
            # --model_name "$MODEL_NAME" \ # Removed, from config
            # --output_dir "$OUTPUT_DIR" \ # Removed, Python script constructs this
            # --batch_size $BATCH_SIZE \ # Removed, from config
            # --num_beams 1 \ # Removed, from config
            # --temperature $TEMPERATURE \ # Removed, from config
            # --max_new_tokens $MAX_TOKENS \ # Removed, from config
            # --layer_indices "$LAYER_INDICES" \ # Removed, from config
            # --prompt_template "$PROMPT_TEMPLATE" \ # Removed, from config
            # --context_mode "sentence" \ # Removed, from config
            # --duplicate_handling "remove" \ # Removed, from config
            # --save_hidden_states \ # Removed, from config
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
            # Iterate through data paths - REMOVED loop, assuming single data source from config
            # for DATA_PATH in "${DATA_PATHS[@]}"
            # do
                # Run task on specified GPU and put in background
                # run_task "$TARGET_WORD" "$METHOD" "$DATA_PATH" "$gpu_id" "$CONTEXT_WINDOW" &
                run_task "$TARGET_WORD" "$METHOD" "$gpu_id" "$CONTEXT_WINDOW" &
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
            # done
        done
    done

    # Wait for all remaining tasks to complete
    wait
done

echo "All processing complete" | tee -a "$LOG_DIR/test.log" 