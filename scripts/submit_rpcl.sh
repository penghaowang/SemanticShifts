#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_rpcl
#SBATCH --nodes=1
#SBATCH --output=logs/submit_rpcl_%j.log  # Use $LOG_DIR
#SBATCH --error=logs/submit_rpcl_error_%j.log # Use $LOG_DIR    
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
words=($WORDS_GROUP1) 

# 定义所有需要处理的提取方法 (Using list from config)
methods=($METHODS_SUBSET1) 

# 要合并的层配置列表 (Using list from config - Bash reads the space-separated quoted strings)
layer_configs=($LAYER_CONFIGS_RPCL)

# 降维方法 (Using list from config)
dimension_methods=($DIMENSION_METHODS_ALL)

# 基础目录（根据实际情况调整） (Using paths from config)
# BASE_LABEL_DIR loaded
# BASE_HS_DIR loaded
OUTPUT_DIR_BASE="combined_layers_plots" # Keep specific base or use $OUTPUT_DIR_BASE

# 创建输出目录 (Using path from config)
mkdir -p "$OUTPUT_DIR_BASE"
mkdir -p "$LOG_DIR"

# 遍历每个词和提取方法
for word in ${words[@]}; do
    # 将冒号替换为下划线，构成目录名（如：import:NOUN -> import_NOUN）
    word_folder=$(echo "$word" | sed 's/:/_/g')
    
    # Note: The original script had 'method' uninitialized here, likely a bug.
    # Assuming loop over methods is intended.
    for method in ${methods[@]}; do
        echo "开始处理 $word (目录：$word_folder) 提取方法: $method"
        # 隐藏状态目录：在 BASE_HS_DIR/word_folder/method/combined 下查找最新的 hidden_states_* 目录
        hs_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/hidden_states_*"
        hs_dir=$(ls -d $hs_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$hs_dir" ]; then
            echo "未找到隐藏状态目录匹配模式: $hs_dir_pattern"
            continue
        fi
        HIDDEN_STATES_DIR="${hs_dir}"
        if [ ! -d "$HIDDEN_STATES_DIR" ]; then
            echo "隐藏状态目录不存在: $HIDDEN_STATES_DIR"
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

        # 对每个层配置和降维方法进行处理
        for layers_config in "${layer_configs[@]}"; do # Iterate over array from config
            # Remove potential outer quotes added by bash for space-separated strings
            layers=$(echo $layers_config | sed "s/^'//;s/'$//")
            for dimension_method in ${dimension_methods[@]}; do # Iterate over space-separated string
                echo "正在处理 $word ($method) - 层配置: $layers - 降维方法: $dimension_method"
                
                # 将层配置中的逗号和连字符替换为下划线，用于文件名
                layer_name=$(echo "$layers" | sed 's/,/_/g' | sed 's/-/to/g')
                
                # 输出目录，包含层配置信息
                OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}_${dimension_method}_layers_${layer_name}"
                mkdir -p "$OUTPUT_DIR"
                
                # 使用 srun 调用 Python 脚本
                srun python ../plot/run_plot_comb_layers.py \
                  --hidden_states_dir "$HIDDEN_STATES_DIR" \
                  --sentence_csv "$SENTENCE_CSV" \
                  --label_csv "$LABEL_CSV" \
                  --output_dir "$OUTPUT_DIR" \
                  --layers "$layers" \
                  --method "$dimension_method" \
                  --word "$word" \
                  --extraction_method "$method"
                
                echo "完成 $word ($method) - 层 $layers - 降维方法 $dimension_method"
            done
        done
    done
done

echo "所有任务完成。" 