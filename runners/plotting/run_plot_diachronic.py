import argparse
import torch
import os
import numpy as np
import logging
import pandas as pd

# 从 evaluate.py 导入刚才写好的函数
from evaluate import merge_label, resolve_file_path, get_time_periods
from plot import plot_hidden_states_layers, fix_tensor_from_series
from logger_config import setup_logger

# 设置日志
logger = setup_logger('run_plot', 'logs/run_plot.log')

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Run plot_hidden_states_layers with hidden states and optional labels")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to hidden states tensor file (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=True, help='Path to the sentence CSV file')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to the label CSV file')
    parser.add_argument('--output_dir', type=str, default='plot_output', help='Where to save the plots')
    parser.add_argument('--method', type=str, default='umap', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='降维方法')
    parser.add_argument('--layers', type=str, required=False, help='要分析的层，例如"0,1,2"或"0-5"')
    parser.add_argument('--word', type=str, required=False, help='目标词')
    parser.add_argument('--extraction_method', type=str, required=False, help='提取方法')
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

    # 检查必要的CSV文件是否存在
    if not os.path.exists(args.sentence_csv):
        logger.error(f"句子CSV文件不存在: {args.sentence_csv}")
        return
    
    if not os.path.exists(args.label_csv):
        logger.error(f"标签CSV文件不存在: {args.label_csv}")
        return

    # 执行 merge_label
    logger.info("开始合并并绘制带标签的散点图。")
    df_merged = merge_label(
        hs_pt_file=hidden_states_file,
        sentence_csv_file=args.sentence_csv,
        label_csv_file=args.label_csv
    )
    # filter out -1 label
    df_merged = df_merged[df_merged['label_index'] != -1]
    if df_merged.empty:
        logger.error("合并结果为空，无法绘图。")
        return

    logger.info(f"合并完成，DataFrame 行数: {len(df_merged)}")
    
    # 获取预定义的时间段
    time_periods = get_time_periods()
    logger.info(f"使用预定义的时间段: {time_periods}")
    
    # 为每个预定义的时间段生成图表
    for period in time_periods:
        # 解析时间段字符串，例如"1990-2000"
        start_year, end_year = map(int, period.split('-'))
        
        # 根据年份筛选数据
        period_df = df_merged[(df_merged['year'] >= start_year) & (df_merged['year'] <= end_year)]
        
        if period_df.empty:
            continue
        
        period_hidden_states = fix_tensor_from_series(period_df["hidden_states"])
        period_labels = period_df["definition"]
        
        # 为每个时间段创建单独的输出目录
        period_dir = os.path.join(args.output_dir, f"period_{period}")
        os.makedirs(period_dir, exist_ok=True)
        
        # 绘制该时间段的图表
        plot_hidden_states_layers(
            period_hidden_states, 
            labels=period_labels, 
            method=args.method, 
            save_dir=period_dir, 
            show_plots=False
        )
    
    logger.info(f"所有时间段的图表已保存到目录: {args.output_dir}")

if __name__ == "__main__":
    main()