#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_rpcl
#SBATCH --nodes=1
#SBATCH --output=logs/submit_rpcl_%j.log  
#SBATCH --error=logs/submit_rpcl_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=a-a05
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00

# 加载必要的模块
module load python gcc cuda

# load your env
# source /iopsstor/scratch/cscs/phwang/.myvenvs/llm/bin/activate

# REMOVED: Set env vars (e.g., HUGGING_FACE_HUB_TOKEN, HF_CACHE_PATH, BASE_STORAGE_PATH) outside the script
# export HUGGING_FACE_HUB_TOKEN=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 # Removed hardcoded token
# export TOKENIZERS_PARALLELISM=false
# export HF_HOME="${HF_CACHE_PATH:-/capstor/scratch/cscs/phwang}" # Use HF_CACHE_PATH env var, fallback to original if not set

# REMOVED: Parameters defined in config YAML
# 设置基本参数
# BASE_HS_DIR="${BASE_STORAGE_PATH:-/iopsstor/scratch/cscs/phwang}/hidden_states" # Use BASE_STORAGE_PATH env var
OUTPUT_DIR="method_correlations" # Keep base output dir for script output
# BATCH_SIZE=100
# FILE_PATTERN="*.pt"
# 定义提取方法列表
# METHODS=("input_last_token" "eos_token" "input_mean" "output_mean" "output_eos")
# 定义要分析的层索引（从0开始）
# LAYER_INDICES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"

# REMOVED: Logic for finding word folders moved to Python script based on config
# echo "正在搜索所有词汇目录..."
# WORD_FOLDERS=()
# for dir in "$BASE_HS_DIR"/*; do
#   if [ -d "$dir" ]; then
#     word_folder=$(basename "$dir")
#     for method in "${METHODS[@]}"; do
#       if [ -d "$dir/$method" ]; then
#         WORD_FOLDERS+=("$word_folder")
#         break
#       fi
#     done
#   fi
# done
# if [ ${#WORD_FOLDERS[@]} -eq 0 ]; then
#   echo "错误: 未找到任何有效的词汇目录"
#   exit 1
# fi

# 打印参数信息 - REMOVED: Parameters are in config
# echo "======================= 提取方法相关性分析 ======================="
# echo "基础隐藏状态目录: $BASE_HS_DIR"
# echo "提取方法列表: ${METHODS[*]}"
# echo "指定层索引: $LAYER_INDICES"
# echo "找到 ${#WORD_FOLDERS[@]} 个词汇目录"
# echo "词汇列表: ${WORD_FOLDERS[*]}"
# echo "批处理大小: $BATCH_SIZE"
# echo "文件匹配模式: $FILE_PATTERN"
# echo "输出目录: $OUTPUT_DIR"
# echo "==============================================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# REMOVED: Layer param construction based on removed variable
# LAYER_PARAM=""
# if [ -n "$LAYER_INDICES" ]; then
#   LAYER_PARAM="--layer_indices $LAYER_INDICES"
# fi

# 运行Python脚本
python ../runners/plotting/run_plot_correlation.py \
  --config "config/correlation_config.yaml" \
  --output_dir "$OUTPUT_DIR"
  # --base_hs_dir "$BASE_HS_DIR" \ # Removed, from config
  # --methods "${METHODS[@]}" \ # Removed, from config
  # --word_folders "${WORD_FOLDERS[@]}" \ # Removed, Python script finds these
  # $LAYER_PARAM \ # Removed, from config
  # --batch_size $BATCH_SIZE \ # Removed, from config
  # --file_pattern "$FILE_PATTERN" # Removed, from config

# 检查执行结果
if [ $? -eq 0 ]; then
  echo "相关性分析成功完成！"
  echo "每个词汇的结果保存在 $OUTPUT_DIR/[词汇名称]/ 目录下"
  echo "所有词汇的综合结果保存在 $OUTPUT_DIR/ 目录下"
  
  # 检查是否生成了总的热图
  if [ -f "$OUTPUT_DIR/all_words_method_correlation_heatmap.png" ]; then
    echo "总热图已生成: $OUTPUT_DIR/all_words_method_correlation_heatmap.png"
  else
    echo "警告: 未生成总热图，可能是因为没有足够的有效数据"
  fi
else
  echo "相关性分析失败，请检查日志文件了解详细信息。"
fi 