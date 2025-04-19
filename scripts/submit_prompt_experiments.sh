#!/bin/bash

# 设置日志目录
LOG_DIR="logs/prompt_experiments"
mkdir -p $LOG_DIR

# 记录实验开始时间
echo "Starting prompt template experiments at $(date)" | tee -a "$LOG_DIR/experiment.log"

# 数据路径
DATA_PATH="processed_data/cs_bulletin_pdf_en_1014_Llama-2-7b_fp16_137666_perplexity_scores.csv"

# 模型名称
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"

# 实验配置
BATCH_SIZE=16
NUM_BEAMS=4
TEMPERATURE=0.7
MAX_NEW_TOKENS=100
LAYER_INDICES="all"

# 运行不同的prompt模板实验
for TEMPLATE in "basic" "context_aware" "multi_sense" "finance" "summary"; do
    echo "Running experiment with template: $TEMPLATE" | tee -a "$LOG_DIR/experiment.log"
    
    # 创建输出目录
    OUTPUT_DIR="outputs/prompt_experiments/${TEMPLATE}"
    mkdir -p $OUTPUT_DIR
    
    # 运行推理
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
        
    # 检查是否成功完成
    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment with template: $TEMPLATE" | tee -a "$LOG_DIR/experiment.log"
    else
        echo "Failed experiment with template: $TEMPLATE" | tee -a "$LOG_DIR/experiment.log"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_DIR/experiment.log"
done

# 记录实验结束时间
echo "Completed all prompt template experiments at $(date)" | tee -a "$LOG_DIR/experiment.log"

# 分析结果
echo "Analyzing results..." | tee -a "$LOG_DIR/experiment.log"

# 可以在这里添加结果分析的代码
# 例如：比较不同模板的性能，生成报告等

echo "Analysis completed." | tee -a "$LOG_DIR/experiment.log" 