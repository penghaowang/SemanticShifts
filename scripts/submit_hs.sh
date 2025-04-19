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

# 设置环境变量
# Line removed by filter-repo due to potential secret
export TOKENIZERS_PARALLELISM=false
# Line removed by filter-repo due to potential secret

# 设置日志目录
LOG_DIR="logs/hidden_states"
mkdir -p $LOG_DIR

# 测试配置
DATA_PATHS=("data/combined.csv")
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# 基础参数
BATCH_SIZE=8
TEMPERATURE=0.3
MAX_TOKENS=50
PROMPT_TEMPLATE="icl_basic"  # 添加为超参数

# 不同的上下文窗口大小
CONTEXT_WINDOWS=(0 1 3 5 10)

# 修改层索引格式，使用逗号分隔而不是空格
LAYER_INDICES="30"

# 目标词列表
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

# 提取方法列表
EXTRACTION_METHODS=("output_mean")

# 定义运行单个任务的函数
run_task() {
    local target_word=$1
    local method=$2
    local data_path=$3
    local gpu_id=$4
    local context_window=$5
    
    local data_file=$(basename "$data_path")
    local data_name="${data_file%.*}"  # 移除扩展名
    
    # 将目标词中的冒号替换为下划线
    local safe_target_word=${target_word//:/_}
    
    echo "=== 任务开始 ===" | tee -a "$LOG_DIR/test.log"
    echo "GPU $gpu_id: ${target_word} - ${method} - ${data_path} - 上下文窗口: ${context_window}" | tee -a "$LOG_DIR/test.log"

    OUTPUT_DIR="/iopsstor/scratch/cscs/phwang/hidden_states_llama38B_cw${context_window}_${PROMPT_TEMPLATE}/${safe_target_word}/${method}/${data_name}"
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
            # 遍历数据路径
            for DATA_PATH in "${DATA_PATHS[@]}"
            do
                # 在指定GPU上运行任务并放入后台
                run_task "$TARGET_WORD" "$METHOD" "$DATA_PATH" "$gpu_id" "$CONTEXT_WINDOW" &
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
            done
        done
    done

    # 等待所有剩余任务完成
    wait
done

echo "全部处理完成" | tee -a "$LOG_DIR/test.log" 