import argparse
import os
import torch
import pandas as pd

# 复用evaluate.py里用于合并 (hidden_states + sentence_csv + label_csv) 的函数
from evaluate import merge_label, resolve_file_path
from plot import fix_tensor_from_series
from plot import plot_all_layers_together_dbscan
from logger_config import setup_logger

# 设置日志
logger = setup_logger('run_dbscan_all_layer', 'logs/run_dbscan_all_layer.log')

def main():
    parser = argparse.ArgumentParser(description="对所有层的隐藏状态合并后用DBSCAN聚类并可视化")
    parser.add_argument('--hidden_states_file', type=str, required=True, 
                        help='隐藏状态文件(.pt)，形状 [N,L,D] 或 [N,L,B,D]')
    parser.add_argument('--sentence_csv', type=str, required=True, 
                        help='句子CSV，用于 merge_label')
    parser.add_argument('--label_csv', type=str, required=False, 
                        help='标签CSV（可选，如果不需要也行）')
    parser.add_argument('--output_dir', type=str, default='all_layers_dbscan_plots', 
                        help='图表保存的目录')
    parser.add_argument('--eps', type=float, default=0.5, 
                        help='DBSCAN的eps参数')
    parser.add_argument('--min_samples', type=int, default=5, 
                        help='DBSCAN的min_samples参数')
    parser.add_argument('--method', type=str, default='umap', 
                        choices=['pca','tsne','umap','zca'],
                        help='降维方法')
    args = parser.parse_args()

    # 找到隐藏状态文件
    hidden_states_file = resolve_file_path(
        args.hidden_states_file,
        "*.pt", 
        f"错误: 隐藏状态文件未找到: {args.hidden_states_file}",
        logger
    )
    
    if hidden_states_file is None:
        return

    # 1) 合并数据
    logger.info("==> 合并 hidden_states 与 CSV ...")
    df_merged = merge_label(
        hs_pt_file=hidden_states_file,
        sentence_csv_file=args.sentence_csv,
        label_csv_file=args.label_csv if args.label_csv else None,
        merge_on="sentence", 
        join_type="inner"
    )
    if df_merged.empty:
        logger.error("合并结果为空，无法进行聚类。")
        return
    
    logger.info(f"合并完成，共 {len(df_merged)} 行。")

    # 2) 提取隐藏状态
    hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
    logger.info(f"hidden_states.shape = {hidden_states.shape}")

    # 3) 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "all_layers_dbscan.png")

    # 4) 调用合并后聚类+可视化函数
    logger.info(f"开始 DBSCAN 聚类 (eps={args.eps}, min_samples={args.min_samples})...")
    plot_all_layers_together_dbscan(
        hidden_states,
        df=df_merged,
        content_column='generated_content',
        eps=args.eps,
        min_samples=args.min_samples,
        method=args.method,
        save_path=out_png,
        show_plot=False,
        top_n=3
    )
    logger.info(f"聚类+可视化完成。图表已保存到: {out_png}")

if __name__ == "__main__":
    main()
