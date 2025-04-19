#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=stacked_layers_plot
#SBATCH --nodes=1
#SBATCH --output=logs/stacked_layers_plot_%j.log  
#SBATCH --error=logs/stacked_layers_plot_error_%j.log     
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
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# 定义目标词（格式：word:POS）和提取方法列表 (Using lists from config)
words=($WORDS_GROUP1 $WORDS_GROUP2_DBSCAN) # Combine groups as used in original script
methods=($METHODS_ALL)

# 基础目录（根据实际情况调整） (Using paths from config)
# BASE_HS_DIR loaded from config
# BASE_LABEL_DIR loaded from config
OUTPUT_DIR_BASE="stacked_layers_plots" # Keep specific output base or use $OUTPUT_DIR_BASE
DIM_REDUCE_METHOD=${DEFAULT_DIM_REDUCE_METHOD:-umap} # Use variable from config

# 创建日志目录 (Using path from config)
mkdir -p "$LOG_DIR"

# 遍历每个词和提取方法
for word in "${words[@]}"; do
    # 将冒号替换为下划线，构成目录名（如：import:NOUN -> import_NOUN）
    word_folder=$(echo "$word" | sed 's/:/_/g')
    
    for method in "${methods[@]}"; do
        echo "开始处理 $word (目录：$word_folder) 提取方法: $method"

        # 隐藏状态文件：在 BASE_HS_DIR/word_folder/method/combined 下查找最新的 hidden_states_* 目录
        hs_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/hidden_states_*"
        hs_dir=$(ls -d $hs_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$hs_dir" ]; then
            echo "未找到隐藏状态目录匹配模式: $hs_dir_pattern"
            continue
        fi
        HIDDEN_STATES_FILE="${hs_dir}/hidden_states_${word_folder}_hidden_states_${method}.pt"
        if [ ! -f "$HIDDEN_STATES_FILE" ]; then
            echo "隐藏状态文件不存在: $HIDDEN_STATES_FILE"
            continue
        fi

        # 句子 CSV 文件：在 BASE_HS_DIR/word_folder/method/combined 下查找最新的 icl_basic_${method}_${word_folder}_* 目录
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

        # 标签 CSV 文件：在 BASE_LABEL_DIR 下查找对应的标签文件
        LABEL_CSV="${BASE_LABEL_DIR}/${word_folder}_labeled.csv"
        if [ ! -f "$LABEL_CSV" ]; then
            echo "标签 CSV 文件不存在: $LABEL_CSV"
            continue
        fi

        # 输出目录：按词和方法保存
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}_${DIM_REDUCE_METHOD}"
        mkdir -p "$OUTPUT_DIR"

        echo "正在为 $word ($method) 生成堆叠层图表..."
        echo "隐藏状态文件: $HIDDEN_STATES_FILE"
        echo "句子 CSV 文件: $SENTENCE_CSV"
        echo "标签 CSV 文件: $LABEL_CSV"
        echo "图表输出目录: $OUTPUT_DIR"

        # 使用 srun 调用 Python 脚本 (Using dim method from config)
        srun python ../plot/run_plot_stacked_layers.py \
          --hidden_states_dir "$HIDDEN_STATES_FILE" \
          --sentence_csv "$SENTENCE_CSV" \
          --label_csv "$LABEL_CSV" \
          --output_dir "$OUTPUT_DIR" \
          --method "$DIM_REDUCE_METHOD"

        echo "处理 $word ($method) 完成。"
    done
done

echo "所有任务完成。" 