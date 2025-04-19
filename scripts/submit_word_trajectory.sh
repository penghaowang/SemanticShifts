#!/bin/bash
#SBATCH --job-name=word_trajectory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/word_trajectory_%j.out
#SBATCH --error=logs/word_trajectory_%j.err
#SBATCH --partition=normal

# 加载必要的模块
module load python/3.9
module load cuda/11.7

# 获取命令行参数
HIDDEN_STATES_DIR=$1  # 隐藏状态目录
SENTENCE_CSV=$2       # 句子CSV文件
LABEL_CSV=$3          # 标签CSV文件
OUTPUT_DIR=$4         # 输出目录
WORD=${5:-""}         # 目标词（可选）
POS=${6:-""}          # 词性（可选）
METHOD=${7:-"umap"}   # 降维方法（默认umap）
LAYER_IDX=${8:--1}    # 层索引（默认-1，表示所有层平均）
TIME_BIN=${9:-"year"} # 时间分组级别（默认按年份）
ARROW_SCALE=${10:-0.1} # 箭头大小（默认0.1）

# 确保日志目录存在
mkdir -p logs

# 打印参数
echo "----------- 任务参数 ------------"
echo "隐藏状态目录: $HIDDEN_STATES_DIR"
echo "句子CSV文件: $SENTENCE_CSV"
echo "标签CSV文件: $LABEL_CSV"
echo "输出目录: $OUTPUT_DIR"
echo "目标词: $WORD"
echo "词性: $POS"
echo "降维方法: $METHOD"
echo "层索引: $LAYER_IDX"
echo "时间分组级别: $TIME_BIN"
echo "箭头大小: $ARROW_SCALE"
echo "---------------------------------"

# 构建参数字符串
ARGS="--mode single --hidden_states_dir $HIDDEN_STATES_DIR --sentence_csv $SENTENCE_CSV --label_csv $LABEL_CSV --output_dir $OUTPUT_DIR --method $METHOD --layer_idx $LAYER_IDX --time_bin_level $TIME_BIN --arrow_scale $ARROW_SCALE"

# 添加可选参数
if [ ! -z "$WORD" ]; then
  ARGS="$ARGS --word $WORD"
fi

if [ ! -z "$POS" ]; then
  ARGS="$ARGS --pos $POS"
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行Python脚本
echo "开始绘制词轨迹图..."
python ../plot/plot_word_trajectory.py $ARGS

echo "任务完成！"

# 使用示例:
# sbatch submit_word_trajectory.sh /path/to/hidden_states path/to/sentences.csv path/to/labels.csv output_dir bank NOUN umap -1 period 0.15 