#!/bin/bash

# Set log directory
LOG_DIR="logs/prompt_experiments"
mkdir -p $LOG_DIR

# Record experiment start time
echo "Starting prompt template experiments at $(date)" | tee -a "$LOG_DIR/experiment.log"

# Data path
DATA_PATH="processed_data/cs_bulletin_pdf_en_1014_Llama-2-7b_fp16_137666_perplexity_scores.csv"

# Model name
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"

# Experiment configuration
BATCH_SIZE=16
NUM_BEAMS=4
TEMPERATURE=0.7
MAX_NEW_TOKENS=100
LAYER_INDICES="all"

# Run experiments with different prompt templates
for TEMPLATE in "basic" "context_aware" "multi_sense" "finance" "summary"; do
    echo "Running experiment with template: $TEMPLATE" | tee -a "$LOG_DIR/experiment.log"
    
    # Create output directory
    OUTPUT_DIR="outputs/prompt_experiments/${TEMPLATE}"
    mkdir -p $OUTPUT_DIR
    
    # Run inference
    python ../run_inference.py \
        --data_path $DATA_PATH \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --num_beams $NUM_BEAMS \
        --temperature $TEMPERATURE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --layer_indices $LAYER_INDICES \
        --prompt_template $TEMPLATE \
        2>&1 | tee "$LOG_DIR/${TEMPLATE}.log"
        
    # Check if completed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment with template: $TEMPLATE" | tee -a "$LOG_DIR/experiment.log"
    else
        echo "Failed experiment with template: $TEMPLATE" | tee -a "$LOG_DIR/experiment.log"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_DIR/experiment.log"
done

# Record experiment end time
echo "Completed all prompt template experiments at $(date)" | tee -a "$LOG_DIR/experiment.log"

# Analyze results
echo "Analyzing results..." | tee -a "$LOG_DIR/experiment.log"

# Analysis code can be added here
# e.g., compare performance of different templates, generate reports, etc.

echo "Analysis completed." | tee -a "$LOG_DIR/experiment.log" 