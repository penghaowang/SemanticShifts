#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=layer_groups_plot
#SBATCH --nodes=1
#SBATCH --output=logs/layer_groups_plot_%j.log  
#SBATCH --error=logs/layer_groups_plot_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=a-a05
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00

# 加载必要的模块
module load python gcc cuda

# 激活虚拟环境
# load your env

# 设置环境变量
# Line removed by filter-repo due to potential secret
export TOKENIZERS_PARALLELISM=false
# Line removed by filter-repo due to potential secret

# 定义目标词（格式：word:POS）和提取方法列表
words=(
    # 第一组词 - 名词
    "market:NOUN" "rate:NOUN" "bank:NOUN" "interest:NOUN" "investment:NOUN"
    "bond:NOUN" "share:NOUN" "capital:NOUN" "exchange:NOUN" "tax:NOUN"
    "growth:NOUN" "security:NOUN" "company:NOUN" "dollar:NOUN" "debt:NOUN"
    "equity:NOUN" "profit:NOUN" "loss:NOUN" "gain:NOUN" "decline:NOUN"
    # 第二组词 - 名词和形容词
    "import:NOUN" "export:NOUN" "money:NOUN" "price:NOUN" "product:NOUN" 
    "sale:NOUN" "agreement:NOUN" "annual:ADJ" "financial:ADJ" "net:ADJ" 
    "industrial:ADJ" "traditional:ADJ" "monetary:ADJ" "inflationary:ADJ" "foreign:ADJ" 
    "public:ADJ" "private:ADJ" "corporate:ADJ" "real:ADJ" "available:ADJ" 
    "strong:ADJ" "stable:ADJ" "fair:ADJ" "competitive:ADJ"
)
methods=("input_last_token" "eos_token" "input_mean" "output_mean" "output_eos")

# 基础目录（根据实际情况调整）
BASE_HS_DIR="/iopsstor/scratch/cscs/phwang/hidden_states"
BASE_LABEL_DIR="/users/phwang/users/master_thesis2/labeled_data_o3_mini"
OUTPUT_DIR_BASE="layer_groups_plots"
DIM_REDUCE_METHOD="umap"  # 可选: 'pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'

# 创建日志目录
mkdir -p logs

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

        echo "正在为 $word ($method) 生成三组层的图表..."
        echo "隐藏状态文件: $HIDDEN_STATES_FILE"
        echo "句子 CSV 文件: $SENTENCE_CSV"
        echo "标签 CSV 文件: $LABEL_CSV"
        echo "图表输出目录: $OUTPUT_DIR"

        # 使用 srun 调用 Python 脚本
        srun python ../plot/run_plot_layer_groups.py \
          --hidden_states_dir "$HIDDEN_STATES_FILE" \
          --sentence_csv "$SENTENCE_CSV" \
          --label_csv "$LABEL_CSV" \
          --output_dir "$OUTPUT_DIR" \
          --method "$DIM_REDUCE_METHOD"

        echo "处理 $word ($method) 完成。"
    done
done

echo "所有任务完成。" 