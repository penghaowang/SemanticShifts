#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_wordtraj
#SBATCH --nodes=1
#SBATCH --output=logs/submit_wordtraj_%j.log  
#SBATCH --error=logs/submit_wordtraj_error_%j.log     
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
words=($WORDS_GROUP1) # Use group1 words from config

# 定义所有需要处理的提取方法 (Using list from config)
methods=($METHODS_SUBSET1) # Use subset from config

# 降维方法 (Using list from config)
dimension_methods=($DIMENSION_METHODS_ALL)

# 时间分组方式 (Using list from config)
time_bin_levels=($TIME_BIN_LEVELS)

# 箭头大小 (Using value from config)
arrow_scale=${DEFAULT_ARROW_SCALE:-0.15}

# 基础目录（根据实际情况调整） (Using paths from config)
# BASE_LABEL_DIR loaded from config
# BASE_HS_DIR loaded from config
OUTPUT_DIR_BASE="word_trajectory_plots" # Keep specific output base or use $OUTPUT_DIR_BASE

# 创建输出目录 (Using path from config)
mkdir -p "$OUTPUT_DIR_BASE"
mkdir -p "$LOG_DIR"

# 遍历每个词和提取方法
for word in ${words[@]}; do # Iterate over space-separated string
    # 提取词和词性
    word_part=$(echo "$word" | cut -d':' -f1)
    pos_part=$(echo "$word" | cut -d':' -f2)
    
    # 将冒号替换为下划线，构成目录名（如：import:NOUN -> import_NOUN）
    word_folder=$(echo "$word" | sed 's/:/_/g')
    
    echo "开始处理 $word (目录：$word_folder)"
    
    for method in ${methods[@]}; do # Iterate over space-separated string
        echo "使用提取方法: $method"
        
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

        # 对每个降维方法和时间分组级别进行处理
        for dim_method in ${dimension_methods[@]}; do # Iterate over space-separated string
            for time_bin in ${time_bin_levels[@]}; do # Iterate over space-separated string
                echo "正在处理 $word ($method) - 降维方法: $dim_method - 时间分组: $time_bin"
                
                # 输出目录，包含所有参数信息
                OUTPUT_DIR="${OUTPUT_DIR_BASE}/${word_folder}/${method}_${dim_method}_${time_bin}"
                mkdir -p "$OUTPUT_DIR"
                
                # 使用 srun 调用 Python 脚本 (Using arrow_scale from config)
                srun python ../plot/plot_word_trajectory.py \
                  --hidden_states_dir "$HIDDEN_STATES_DIR" \
                  --sentence_csv "$SENTENCE_CSV" \
                  --label_csv "$LABEL_CSV" \
                  --output_dir "$OUTPUT_DIR" \
                  --method "$dim_method" \
                  --word "$word_part" \
                  --pos "$pos_part" \
                  --arrow_scale "$arrow_scale" \
                  --time_bin_level "$time_bin"
                
                echo "完成 $word ($method) - 降维方法 $dim_method - 时间分组 $time_bin"
            done
        done
    done
done

echo "所有单词的轨迹图生成任务完成。" 