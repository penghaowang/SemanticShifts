#!/bin/bash
#SBATCH --job-name=all_layers_dbscan
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=a-a05
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=logs/all_layers_dbscan_%j.log
#SBATCH --error=logs/all_layers_dbscan_error_%j.log

module load python gcc cuda
# load your env

# Line removed by filter-repo due to potential secret
export TOKENIZERS_PARALLELISM=false
# Line removed by filter-repo due to potential secret

# 你要处理的单词和提取方法
words=("import:NOUN" "export:NOUN")
methods=("input_last_token" "output_eos")

BASE_HS_DIR="/iopsstor/scratch/cscs/phwang/hidden_states"
BASE_LABEL_DIR="/users/phwang/users/master_thesis2/labeled_data_o3_mini"
DIM_REDUCE_METHOD="umap"   # 可改成 pca / tsne / umap / zca
OUTPUT_DIR_ROOT="plots_dbscan_all_layers"

for word in "${words[@]}"; do
    word_folder=$(echo "$word" | sed 's/:/_/g')

    for method in "${methods[@]}"; do
        echo "处理 $word ($method)..."

        hs_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/hidden_states_*"
        hs_dir=$(ls -d $hs_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$hs_dir" ]; then
            echo "未找到隐藏状态目录: $hs_dir_pattern"
            continue
        fi
        HIDDEN_STATES_FILE="${hs_dir}/hidden_states_${word_folder}_hidden_states_${method}.pt"
        if [ ! -f "$HIDDEN_STATES_FILE" ]; then
            echo "隐藏状态文件不存在: $HIDDEN_STATES_FILE"
            continue
        fi

        sentence_dir_pattern="${BASE_HS_DIR}/${word_folder}/${method}/combined/icl_basic_${method}_${word_folder}_*"
        sentence_dir=$(ls -d $sentence_dir_pattern 2>/dev/null | sort | tail -n 1)
        if [ -z "$sentence_dir" ]; then
            echo "未找到句子CSV目录: $sentence_dir_pattern"
            continue
        fi
        SENTENCE_CSV="${sentence_dir}/icl_basic_${method}_${word_folder}.csv"
        if [ ! -f "$SENTENCE_CSV" ]; then
            echo "句子CSV不存在: $SENTENCE_CSV"
            continue
        fi

        LABEL_CSV="${BASE_LABEL_DIR}/${word_folder}_labeled.csv"
        if [ ! -f "$LABEL_CSV" ]; then
            echo "标签CSV不存在: $LABEL_CSV"
            continue
        fi

        # 输出目录
        out_dir="${OUTPUT_DIR_ROOT}/${word_folder}/${method}_all_layers_${DIM_REDUCE_METHOD}"
        mkdir -p "$out_dir"

        echo "==> srun python run_dbscan_all_layer.py"
        srun python ../run_dbscan_all_layer.py \
            --hidden_states_file "$HIDDEN_STATES_FILE" \
            --sentence_csv "$SENTENCE_CSV" \
            --label_csv "$LABEL_CSV" \
            --output_dir "$out_dir" \
            --eps 0.5 \
            --min_samples 5 \
            --method "$DIM_REDUCE_METHOD"

        echo "完成 $word ($method)"
        echo
    done
done

echo "所有任务完成。"
