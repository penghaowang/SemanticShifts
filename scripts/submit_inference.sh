#!/bin/bash

#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_hs
#SBATCH --nodes=1
#SBATCH --output=logs/submit_hs_%j.log  
#SBATCH --error=logs/submit_hs_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=a-a05
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=01:30:00

module load python gcc cuda
# load your env
# source /iopsstor/scratch/cscs/phwang/.myvenvs/llm/bin/activate

# REMOVED: Set env vars (e.g., HUGGING_FACE_HUB_TOKEN, HF_CACHE_PATH, BASE_STORAGE_PATH) outside the script
# export HUGGING_FACE_HUB_TOKEN=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 # Removed hardcoded token
# export TOKENIZERS_PARALLELISM=false
# export HF_HOME="${HF_CACHE_PATH:-/capstor/scratch/cscs/phwang}" # Use HF_CACHE_PATH env var, fallback to original if not set

# 设置日志目录
LOG_DIR="logs/hidden_states"
mkdir -p $LOG_DIR

# REMOVED: Parameters defined in config YAML
# 测试配置
# DATA_PATHS=("data/combined.csv")
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# 基础参数
# BATCH_SIZE=8
# TEMPERATURE=0.3
# MAX_TOKENS=50
# PROMPT_TEMPLATE="icl_basic"  # 添加为超参数
# 不同的上下文窗口大小
# CONTEXT_WINDOWS=(0 1 3 5 10)
# 修改层索引格式，使用逗号分隔而不是空格
# LAYER_INDICES="30"
# 提取方法列表
# EXTRACTION_METHODS=("output_mean")

# 目标词列表 - Kept here as it drives the main loop, could also be moved to config
words=("market:NOUN" "rate:NOUN" "fair:ADJ")

# Define context windows and methods here, or move to config
CONTEXT_WINDOWS=(0 1 3 5 10)
EXTRACTION_METHODS=("output_mean")

# 定义运行单个任务的函数
run_task() {
    local target_word=$1
    local method=$2
    # local data_path=$3 # Removed, assume from config
    local gpu_id=$4
    local context_window=$5
    
    # local data_file=$(basename "$data_path") # Removed, assume from config
    # local data_name="${data_file%.*}"  # 移除扩展名
    local data_name="combined" # Placeholder, Python script should determine input data based on config

    # 将目标词中的冒号替换为下划线
    local safe_target_word=${target_word//:/_}
    
    echo "=== 任务开始 ===" | tee -a "$LOG_DIR/test.log"
    # echo "GPU $gpu_id: ${target_word} - ${method} - ${data_path} - 上下文窗口: ${context_window}" | tee -a "$LOG_DIR/test.log" # Removed data_path
    echo "GPU $gpu_id: ${target_word} - ${method} - 上下文窗口: ${context_window}" | tee -a "$LOG_DIR/test.log"

    # REMOVED: Output dir construction moved to Python script based on config
    # OUTPUT_DIR="${BASE_STORAGE_PATH:-/iopsstor/scratch/cscs/phwang}/hidden_states_llama38B_cw${context_window}_${PROMPT_TEMPLATE}/${safe_target_word}/${method}/${data_name}" # Use BASE_STORAGE_PATH env var
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

    # 检查运行状态
    if [ $? -eq 0 ]; then
        echo "✓ 任务完成" | tee -a "$LOG_DIR/test.log"
    else:
        echo "✗ 任务失败" | tee -a "$LOG_DIR/test.log"
    fi

    echo "=============" | tee -a "$LOG_DIR/test.log"
}

# 遍历目标词
for TARGET_WORD in "${words[@]}"
do
    echo "开始处理: ${TARGET_WORD}" | tee -a "$LOG_DIR/test.log"
    echo "=============" | tee -a "$LOG_DIR/test.log"

    # 使用数组来跟踪后台进程
    declare -a pids
    gpu_id=0

    # 遍历上下文窗口大小
    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"
    do
        # 遍历提取方法
        for METHOD in "${EXTRACTION_METHODS[@]}"
        do
            # 遍历数据路径 - REMOVED loop, assuming single data source from config
            # for DATA_PATH in "${DATA_PATHS[@]}"
            # do
                # 在指定GPU上运行任务并放入后台
                # run_task "$TARGET_WORD" "$METHOD" "$DATA_PATH" "$gpu_id" "$CONTEXT_WINDOW" &
                run_task "$TARGET_WORD" "$METHOD" "$gpu_id" "$CONTEXT_WINDOW" &
                pids+=($!)

                # 更新GPU ID，循环使用0-3
                gpu_id=$(( (gpu_id + 1) % 4 ))

                # 如果已经启动了4个任务，等待其中一个完成
                if [ ${#pids[@]} -eq 4 ]; then
                    wait -n  # 等待任意一个子进程完成
                    # 移除已完成的进程ID
                    for pid in "${pids[@]}"; do
                        if ! kill -0 $pid 2>/dev/null; then
                            pids=("${pids[@]/$pid}")
                        fi
                    done
                fi
            # done
        done
    done

    # 等待所有剩余任务完成
    wait
done

echo "全部处理完成" | tee -a "$LOG_DIR/test.log" 