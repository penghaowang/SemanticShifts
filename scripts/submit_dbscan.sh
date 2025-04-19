#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_dbscan
#SBATCH --nodes=1
#SBATCH --output=logs/submit_dbscan_%j.log  
#SBATCH --error=logs/submit_dbscan_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=a-a05
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00

# Source the config loader and load variables
source ../scripts/load_config.sh
load_yaml_config ../config.yaml

# 加载必要的模块
module load python gcc cuda

# 激活虚拟环境 (Using path from config)
# load your env

# 设置环境变量 (Using paths/settings from config)
# Note: HUGGING_FACE_HUB_TOKEN should be set externally
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME="$HF_HOME"

# 定义目标词（格式：word:POS）和提取方法列表 (Using lists from config)
words=($WORDS_GROUP2_DBSCAN)
methods=($METHODS_ALL)

# 基础目录（根据实际情况调整） (Using paths from config)
# BASE_HS_DIR loaded
OUTPUT_DIR_BASE="dbscan_plots" # Keep specific base or use $OUTPUT_DIR_BASE
DIM_REDUCE_METHOD=${DEFAULT_DIM_REDUCE_METHOD:-umap} # Use variable from config
DBSCAN_EPS=${DBSCAN_EPS:-0.5}
DBSCAN_MIN_SAMPLES=${DBSCAN_MIN_SAMPLES:-5}

# 创建日志目录 (Using path from config)
mkdir -p "$LOG_DIR"

# 遍历每个词和提取方法
for word in ${words[@]}; do
    # 将冒号替换为下划线，构成目录名（如：import:NOUN -> import_NOUN）
    word_folder=$(echo "$word" | sed 's/:/_/g')
    
    for method in ${methods[@]}; do
        echo "开始处理 $word (目录：$word_folder) 提取方法: $method"

        # 隐藏状态文件：在 BASE_HS_DIR/word_folder/method/combined 下查找最新的 hidden_states_* 目录，
        hs_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/hidden_states_*"
        hs_dir=$(ls -d $hs_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$hs_dir" ]; then
            echo "未找到隐藏状态目录匹配模式: $hs_dir_pattern"
            continue
        fi
        # Variable name corrected as per original script logic
        HIDDEN_STATES_FILE_OR_DIR="${hs_dir}/hidden_states_${word_folder}_hidden_states_${method}.pt"
        # Check if it's a file, if not, assume it's the directory itself
        if [ ! -f "$HIDDEN_STATES_FILE_OR_DIR" ]; then 
             if ls "${hs_dir}"/*.pt > /dev/null 2>&1; then
                 HIDDEN_STATES_FILE_OR_DIR="$hs_dir"
             else
                 echo "隐藏状态文件或目录不存在: ${hs_dir}/hidden_states...pt or .pt files in ${hs_dir}"
                 continue
             fi
        fi

        # 句子 CSV 文件：在 BASE_HS_DIR/word_folder/method/combined 下查找最新的 icl_basic_${method}_${word_folder}_* 目录，
        sentence_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/icl_basic_${method}_${word_folder}_*"
        sentence_dir=$(ls -d $sentence_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$sentence_dir" ]; then
            echo "未找到句子 CSV 目录匹配模式: $sentence_dir_pattern"
            continue
        fi
        SENTENCE_CSV="${sentence_dir}/icl_basic_${method}_${word_folder}.csv"
        if [ ! -f "$SENTENCE_CSV" ]; then
            echo "句子 CSV 文件不存在: $SENTENCE_CSV"
            continue
        fi

        # 输出目录：按词和方法保存
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}"
        mkdir -p "$OUTPUT_DIR"

        echo "正在为 $word ($method) 执行DBSCAN聚类分析..."
        echo "隐藏状态文件/目录: $HIDDEN_STATES_FILE_OR_DIR"
        echo "句子 CSV 文件: $SENTENCE_CSV"
        echo "图表输出目录: $OUTPUT_DIR"

        # 使用 srun 调用 Python 脚本 (Using params from config)
        srun python ../run_dbscan.py \
          --hidden_states_dir "$HIDDEN_STATES_FILE_OR_DIR" \
          --sentence_csv "$SENTENCE_CSV" \
          --output_dir "$OUTPUT_DIR" \
          --eps $DBSCAN_EPS \
          --min_samples $DBSCAN_MIN_SAMPLES \
          --method "$DIM_REDUCE_METHOD"

        echo "处理 $word ($method) 完成。"
    done
done

echo "所有任务完成。"
