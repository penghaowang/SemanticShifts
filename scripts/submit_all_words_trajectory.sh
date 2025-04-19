#!/bin/bash
#SBATCH --view=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --job-name=all_words_traj
#SBATCH --nodes=1
#SBATCH --output=logs/all_words_traj_%j.log  
#SBATCH --error=logs/all_words_traj_error_%j.log     
#SBATCH --ntasks=4
#SBATCH --account=a-a05
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00
# 先整合所有时间点的高维向量并统一降维，再提取坐标绘制变迁图。这种方法充分利用了UMAP的全局结构保留能力，确保语义变化的时空连续性在低维空间中准确呈现。
# 加载必要的模块
module load python gcc cuda

# 激活虚拟环境
# load your env

# 设置环境变量
# Line removed by filter-repo due to potential secret
export TOKENIZERS_PARALLELISM=false
# Line removed by filter-repo due to potential secret

# 基础目录
BASE_HS_DIR="/iopsstor/scratch/cscs/phwang/hidden_states"
OUTPUT_DIR_BASE="all_words_trajectory_plots"

# 降维方法
dimension_method="umap"

# 提取方法
extraction_methods=("input_last_token" "eos_token")

# 箭头大小
arrow_scale=0.15

# 金融领域的核心词
word_list=(
    "market" "rate" "bank" "interest" "investment"
    "bond" "share" "capital" "exchange" "tax"
    "growth" "security" "company" "dollar" "debt"
    "equity" "profit" "loss" "gain" "decline"
    "import" "export" "money" "price" "product"
    "sale" "agreement" "annual" "financial" "net"
    "industrial" "traditional" "monetary" "inflationary" "foreign"
    "public" "private" "corporate" "real" "available"
    "strong" "stable" "fair" "competitive"
)

# 创建输出目录
mkdir -p "$OUTPUT_DIR_BASE"
mkdir -p logs

# 检查Python脚本是否存在
if [ ! -f "../plot/plot_word_trajectory.py" ]; then
    echo "错误: plot_word_trajectory.py 文件不存在"
    exit 1
fi

# 对每个提取方法运行一次
for extraction_method in "${extraction_methods[@]}"; do
    echo "开始处理 - 提取方法: $extraction_method - 降维方法: $dimension_method"
    
    # 输出目录，包含参数信息
    output_dir="${OUTPUT_DIR_BASE}/${extraction_method}_${dimension_method}"
    mkdir -p "$output_dir"
    
    # 构建单词列表参数
    word_list_args=""
    for word in "${word_list[@]}"; do
        word_list_args+="$word "
    done
    
    # 使用srun运行Python脚本
    srun python ../plot/plot_word_trajectory.py \
        --base_dir "$BASE_HS_DIR" \
        --output_dir "$output_dir" \
        --method "$dimension_method" \
        --extraction_method "$extraction_method" \
        --arrow_scale "$arrow_scale" \
        --word_list $word_list_args
    
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "错误: 处理 $extraction_method 时出错"
        continue
    fi
    
    echo "完成 - 提取方法: $extraction_method - 降维方法: $dimension_method"
done

echo "所有词汇轨迹图绘制任务完成。" 