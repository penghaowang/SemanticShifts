#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=submit_layer_plot
#SBATCH --nodes=1
#SBATCH --output=logs/submit_layer_plot_%j.log  
#SBATCH --error=logs/submit_layer_plot_error_%j.log     
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


# 定义提取方法列表 (Using list from config)
methods=($METHODS_ALL)

# 基础目录（根据实际情况调整） (Using paths from config)
# BASE_HS_DIR loaded
OUTPUT_DIR_BASE="layer_plots" # Keep specific base or use $OUTPUT_DIR_BASE

# 创建日志目录 (Using path from config)
mkdir -p "$LOG_DIR"

# 遍历每个提取方法
for method in ${methods[@]}; do
    echo "开始处理提取方法: $method"
    
    # 创建输出目录
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${method}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "正在为提取方法 $method 执行层间相似度分析..."
    
    # 运行层间相似度分析脚本
    python ../plot/run_layer_plot.py \
        --base_hs_dir "$BASE_HS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --method "$method"
    
    echo "完成提取方法 $method 的层间相似度分析，结果保存在 $OUTPUT_DIR"
done

echo "所有提取方法的层间相似度分析已完成"
