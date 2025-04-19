#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --output=logs/create_data_%j.log
#SBATCH --error=logs/create_data_error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# 加载环境
module load cray/23.12
module load gcc-native/12.3
module load cray-python/3.11.5
module load cudatoolkit/24.3_12.3

# 设置环境变量
# Line removed by filter-repo due to potential secret
export TOKENIZERS_PARALLELISM=false

# 创建日志目录
mkdir -p logs

# 定义数据集路径
DATASETS=(
    "processed_data/cs_bulletin_pdf_en_1014_Llama-3.1-8B_fp16_137666_perplexity_scores.csv"
    "processed_data/cs_bulletin_ocr_en_1009_Llama-3.1-8B_fp16_423396_perplexity_scores.csv"
)

# 定义context window范围
CONTEXT_WINDOWS=(5 10 20)

# 定义基础输出目录
BASE_OUTPUT_DIR="/capstor/scratch/cscs/phwang/datasets/word_datasets"

# 循环处理不同的context window
for window in "${CONTEXT_WINDOWS[@]}"; do
    # 为每个context window创建单独的输出目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/window_${window}"
    echo "处理 context window = ${window}"
    
    # 运行数据处理脚本
    srun python ../create_data.py \
        --data_paths "${DATASETS[@]}" \
        --model_name "meta-llama/Llama-2-7b-chat-hf" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 32 \
        --max_length 2048 \
        --context_window $window

    # 检查运行状态
    if [ $? -ne 0 ]; then
        echo "处理 context window = ${window} 时失败"
        exit 1
    fi
done

echo "所有context window处理完成"

# 统计生成的数据集
echo "生成的数据集统计："
for window in "${CONTEXT_WINDOWS[@]}"; do
    echo "Context Window = ${window}:"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/window_${window}"
    for dir in "$OUTPUT_DIR"/*; do
        if [ -d "$dir" ]; then
            word_pos=$(basename "$dir")
            num_files=$(find "$dir" -type f | wc -l)
            echo "  $word_pos: $num_files files"
        fi
    done
done 