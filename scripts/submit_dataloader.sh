#!/bin/bash
#SBATCH --job-name=run_dataloader
#SBATCH --output=logs/run_dataloader_%j.log  
#SBATCH --error=logs/run_dataloader_error_%j.log             
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1              
#SBATCH --time=24:00:00                 
#SBATCH --partition=normal   
#SBATCH --constraint=gpu

# Source the config loader and load variables
source ../scripts/load_config.sh
load_yaml_config ../config.yaml

# 加载必要的模块
module load cray/23.12
module load gcc-native/12.3
module load cray-python/3.11.5
module load cudatoolkit/24.3_12.3
# load your env

# 设置环境变量 (Using paths/settings from config)
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# 创建日志目录 (Using path from config)
mkdir -p "$LOG_DIR"

# 定义数据集路径 (Using paths from config)
DATASETS=($DATA_PATHS_CREATE_DATA) # Assuming these are the correct datasets for this script too

# 定义目标词（使用空格分隔）(Using list from config)
TARGET_WORDS="$WORDS_DATALOADER" # Use comma-separated string from config

# 定义上下文窗口大小 (Keep specific values here or move to config if shared)
CONTEXT_WINDOWS=(0 1 2)

# 循环处理每个上下文窗口大小
for window in "${CONTEXT_WINDOWS[@]}"; do
    echo "Processing context window size: $window"
    
    # 为每个窗口大小创建单独的输出目录 (Using base path from config)
    WINDOW_OUTPUT_DIR="/capstor/scratch/cscs/phwang/datasets/window_${window}" # Keep specific path or use $OUTPUT_DIR_BASE
    mkdir -p "$WINDOW_OUTPUT_DIR"
    
    # Run dataloader script (Using model name & batch size from config)
    srun python ../run_dataloader.py \
        --dataset_paths "${DATASETS[@]}" \
        --model_name "$MODEL_NAME_LLAMA3_8B_INSTRUCT" \
        --target_words "$TARGET_WORDS" \
        --batch_size ${DEFAULT_BATCH_SIZE:-32} \
        --max_length 2048 \
        --num_workers 4 \
        --output_dir "$WINDOW_OUTPUT_DIR" \
        --context_mode "sentence" \
        --context_window $window \
        --duplicate_handling "remove" \
        --simple_filter True
        
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Error processing context window size: $window"
        exit 1
    fi
    
    echo "Completed processing for window size: $window"
done

echo "All processing completed successfully"