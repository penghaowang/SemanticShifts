#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=semantic_shift
#SBATCH --nodes=1
#SBATCH --output=logs/semantic_shift_%j.log  
#SBATCH --error=logs/semantic_shift_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=a-a05
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=01:00:00
# 此脚本用于批量处理词汇的语义变化分析，绘制JS散度和轮廓系数变化图

# 加载必要的模块
module load python gcc cuda

# load your env
# source /iopsstor/scratch/cscs/phwang/.myvenvs/llm/bin/activate

# 设置环境变量
# export HUGGING_FACE_HUB_TOKEN=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 # Removed hardcoded token
# export TOKENIZERS_PARALLELISM=false
# export HF_HOME="${HF_CACHE_PATH:-/capstor/scratch/cscs/phwang}" # Use HF_CACHE_PATH env var, fallback to original if not set

# 基础参数
# BASE_DIR="${BASE_STORAGE_PATH:-/iopsstor/scratch/cscs/phwang}/hidden_states" # Use BASE_STORAGE_PATH env var
OUTPUT_BASE_DIR="semantic_shift_plots"
# METHOD="umap"
# LAYER_IDX=-1
# TIME_BIN_LEVEL="period"  # 可选 "year" 或 "period"
# EXTRACTION_METHOD="input_last_token"  # 与plot_word_trajectory.py保持一致

# 金融领域的核心词及其词性
declare -A word_pos
# 使用file_context_0中的词汇列表
for word in "market" "rate" "bank" "interest" "investment" "bond" "share" "capital" "exchange" "tax" "growth" "security" "company" "dollar" "debt" "equity" "gain" "decline" "import" "export" "money" "price" "product" "sale" "agreement" "annual" "financial" "net" "industrial" "traditional" "monetary" "inflationary" "foreign" "public" "private" "corporate" "real" "available" "strong" "stable" "fair" "competitive" "profit" "loss"; do
    # 默认设置为名词，特定形容词单独处理
    word_pos[$word]="NOUN"
    
    # 形容词列表处理
    if [[ "$word" == "financial" || "$word" == "net" || "$word" == "industrial" || "$word" == "traditional" || "$word" == "monetary" || "$word" == "inflationary" || "$word" == "foreign" || "$word" == "public" || "$word" == "private" || "$word" == "corporate" || "$word" == "real" || "$word" == "available" || "$word" == "strong" || "$word" == "stable" || "$word" == "fair" || "$word" == "competitive" || "$word" == "annual" ]]; then
        word_pos[$word]="ADJ"
    fi
done

# 创建输出目录和日志目录
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p logs

# 检查Python脚本是否存在
if [ ! -f "runners/plotting/run_shift_plot.py" ]; then
    echo "错误: run_shift_plot.py 文件不存在"
    exit 1
fi

# echo "开始处理词汇语义变化分析 - 方法: $METHOD - 时间分组: $TIME_BIN_LEVEL"

# 处理每个词汇
for word in "${!word_pos[@]}"; do
    pos=${word_pos[$word]}
    echo "处理词汇: $word (词性: $pos)"
    
    # 去除词汇中的引号（如果有）
    clean_word=$(echo $word | tr -d '"')
    
    # 为每个词创建输出目录
    word_output_dir="${OUTPUT_BASE_DIR}/${clean_word}"
    mkdir -p "$word_output_dir"
    
    # 构建词汇的隐藏状态目录路径
    # WORD_DIR="${BASE_DIR}/${clean_word}_${pos}"
    
    # 隐藏状态目录
    # HS_DIR_PATTERN="${WORD_DIR}/${EXTRACTION_METHOD}/combined/hidden_states_*"
    # HS_DIRS=$(ls -d $HS_DIR_PATTERN 2>/dev/null)
    
    # 使用最新的隐藏状态目录
    # HS_DIR=$(echo $HS_DIRS | tr ' ' '\n' | sort | tail -n 1)
    
    # 查找隐藏状态文件
    # PT_FILES=$(ls ${HS_DIR}/*.pt 2>/dev/null)
    # if [ -z "$PT_FILES" ]; then
    #     echo "警告: 在${HS_DIR}中未找到.pt文件"
    #     continue
    # fi
    
    # 使用第一个pt文件
    # HS_FILE=$(echo $PT_FILES | tr ' ' '\n' | head -n 1)
    
    # 句子CSV文件目录
    # SENTENCE_DIR_PATTERN="${WORD_DIR}/${EXTRACTION_METHOD}/combined/icl_basic_${EXTRACTION_METHOD}_${clean_word}_${pos}_*"
    # SENTENCE_DIRS=$(ls -d $SENTENCE_DIR_PATTERN 2>/dev/null)
    
    # 使用最新的句子目录
    # SENTENCE_DIR=$(echo $SENTENCE_DIRS | tr ' ' '\n' | sort | tail -n 1)
    # SENTENCE_CSV="${SENTENCE_DIR}/icl_basic_${EXTRACTION_METHOD}_${clean_word}_${pos}.csv"
    
    # 标签CSV文件
    # LABEL_CSV="/users/phwang/users/master_thesis2/labeled_data_o3_mini/${clean_word}_${pos}_labeled.csv"
    
    # 运行Python脚本
    srun python runners/plotting/run_shift_plot.py \
        --config "config/plot_config.yaml" \
        --word "$clean_word" \
        --pos "$pos" \
        --output_dir "$word_output_dir"
    
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "错误: 处理词汇 $clean_word 时出错"
        continue
    fi
    
    echo "完成词汇: $clean_word"
done

echo "所有词汇的语义变化分析任务完成。" 

# 收集和汇总所有词语的JSD和熵值数据
echo "开始收集和汇总所有词语的语义变化数据..."

# 检查collect_semantic_data.py是否存在
if [ ! -f "collect_semantic_data.py" ]; then
    echo "错误: collect_semantic_data.py 文件不存在，无法收集汇总数据"
    exit 1
fi

# 运行数据收集脚本
srun python collect_semantic_data.py --config "config/collect_config.yaml"

# 检查脚本运行状态
if [ $? -eq 0 ]; then
    echo "数据收集和汇总成功完成。汇总数据保存在 semantic_shift_summary 目录中。"
else
    echo "错误: 数据收集和汇总过程中出现问题"
    exit 1
fi

echo "整个分析流程已完成。" 