import argparse
import torch
import os
import numpy as np
import logging

# 从 evaluate.py 导入刚才写好的函数
from evaluate import merge_label, resolve_file_path
from plot import plot_hidden_states_layers, fix_tensor_from_series
from logger_config import setup_logger

# 设置日志
logger = setup_logger('run_plot', 'logs/run_plot.log')

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Run plot_hidden_states_layers with hidden states and optional labels")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to hidden states tensor file (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=False, help='Path to the sentence CSV file')
    parser.add_argument('--label_csv', type=str, required=False, help='Path to the label CSV file')
    parser.add_argument('--output_dir', type=str, default='plot_output', help='Where to save the plots')
    parser.add_argument('--method', type=str, default='zca_pca', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='降维方法')
    args = parser.parse_args()

    # 查找隐藏状态文件
    hidden_states_file = resolve_file_path(
        args.hidden_states_dir,
        "*.pt", 
        f"未找到隐藏状态文件: {args.hidden_states_dir}",
        logger
    )
    
    if hidden_states_file is None:
        return

    # 如果没有提供 sentence_csv 和 label_csv，则仅绘制隐藏状态(不区分标签)
    if not args.sentence_csv or not args.label_csv:
        logger.info("未提供 sentence_csv 和 label_csv，将只绘制未带标签的散点图。")
        hidden_states = torch.load(hidden_states_file)
        logger.info(f"已加载隐藏状态，形状为: {hidden_states.shape}")

        os.makedirs(args.output_dir, exist_ok=True)
        plot_hidden_states_layers(
            hidden_states, 
            labels=None,  # 不区分颜色
            method=args.method,
            save_dir=args.output_dir, 
            show_plots=False
        )
        logger.info(f"图表已保存到目录: {args.output_dir}")
    else:
        # 如果提供了 sentence_csv 和 label_csv，则执行 merge_label
        logger.info("提供了 sentence_csv 和 label_csv，开始合并并绘制带标签的散点图。")
        df_merged = merge_label(
            hs_pt_file=hidden_states_file,
            sentence_csv_file=args.sentence_csv,
            label_csv_file=args.label_csv
        )
        if df_merged.empty:
            logger.error("合并结果为空，无法绘图。")
            return

        logger.info(f"合并完成，DataFrame 行数: {len(df_merged)}")
        # 从df_merged中提取隐藏状态和标签，使用fix_tensor_from_series避免警告
        hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
        logger.debug(f"hidden_states shape: {hidden_states.shape}")
        labels = df_merged["definition"]  # 假设标签列名就是"label"，请根据实际修改

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 绘制
        plot_hidden_states_layers(
            hidden_states, 
            labels=labels, 
            method='pca', 
            save_dir=args.output_dir, 
            show_plots=False
        )
        logger.info(f"图表已保存到目录: {args.output_dir}")

if __name__ == "__main__":
    main()