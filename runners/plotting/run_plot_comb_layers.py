import argparse
import torch
import os
import numpy as np
import logging
import pandas as pd
import traceback

# 从 evaluate.py 导入刚才写好的函数
from evaluate import merge_label, resolve_file_path, get_time_periods
from plot import plot_hidden_states_layers, fix_tensor_from_series, plot_all_layers_together_dbscan, plot_hidden_states
from logger_config import setup_logger

# 设置日志
logger = setup_logger('run_plot_comb_layers', 'logs/run_plot_comb_layers.log')

def parse_layers_str(layers_str):
    """
    解析层字符串，支持逗号分隔和范围表示法
    例如: "0,1,2" 或 "0-5" 或 "0,1,5-7"
    
    参数:
        layers_str (str): 层字符串
        
    返回:
        list: 层索引列表
    """
    if not layers_str:
        return []
    
    layers = []
    parts = layers_str.split(',')
    
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    
    return sorted(list(set(layers)))  # 去重并排序

def main():
    try:
        # 定义命令行参数
        parser = argparse.ArgumentParser(description="Run plot_hidden_states_layers with hidden states and optional labels")
        parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to hidden states tensor file (.pt)')
        parser.add_argument('--sentence_csv', type=str, required=True, help='Path to the sentence CSV file')
        parser.add_argument('--label_csv', type=str, required=True, help='Path to the label CSV file')
        parser.add_argument('--output_dir', type=str, default='plot_output', help='Where to save the plots')
        parser.add_argument('--method', type=str, default='umap', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='降维方法')
        parser.add_argument('--layers', type=str, required=True, help='要分析的层，例如"0,1,2"或"0-5"')
        parser.add_argument('--word', type=str, required=False, help='目标词')
        parser.add_argument('--extraction_method', type=str, required=False, help='提取方法')
        args = parser.parse_args()

        logger.info(f"开始处理：词={args.word}, 提取方法={args.extraction_method}, 层={args.layers}, 降维方法={args.method}")
        
        # 查找隐藏状态文件
        hidden_states_file = resolve_file_path(
            args.hidden_states_dir,
            "*.pt", 
            f"未找到隐藏状态文件: {args.hidden_states_dir}",
            logger
        )
        
        if hidden_states_file is None:
            logger.error(f"未找到隐藏状态文件：{args.hidden_states_dir}")
            return 1

        # 检查必要的CSV文件是否存在
        if not os.path.exists(args.sentence_csv):
            logger.error(f"句子CSV文件不存在: {args.sentence_csv}")
            return 1
        
        if not os.path.exists(args.label_csv):
            logger.error(f"标签CSV文件不存在: {args.label_csv}")
            return 1

        # 执行 merge_label
        logger.info("开始合并标签数据。")
        df_merged = merge_label(
            hs_pt_file=hidden_states_file,
            sentence_csv_file=args.sentence_csv,
            label_csv_file=args.label_csv
        )
        
        logger.info(f"合并前数据量: {len(df_merged)}, 标签为-1的行数: {sum(df_merged['label_index'] == -1)}")
        
        # filter out -1 and -3 label
        df_merged = df_merged[df_merged['label_index'] != -1]
        df_merged = df_merged[df_merged['label_index'] != -3]

        
        logger.info(f"过滤label=-1和label=-3后的数据量: {len(df_merged)}")
        
        if df_merged.empty:
            logger.error("合并结果为空，无法绘图。请检查标签数据。")
            return 1

        logger.info(f"合并完成，DataFrame 行数: {len(df_merged)}")
        
        # 分析标签分布
        label_counts = df_merged['definition'].value_counts()
        logger.info(f"标签分布: {label_counts.to_dict()}")
        
        # 解析要分析的层
        layers = parse_layers_str(args.layers)
        if not layers:
            logger.error(f"未能解析层参数: {args.layers}")
            return 1
        
        logger.info(f"将分析以下层: {layers}")
        
        # 获取隐藏状态
        try:
            hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
            logger.info(f"隐藏状态张量形状: {hidden_states.shape}")
        except Exception as e:
            logger.error(f"处理隐藏状态失败: {str(e)}")
            return 1
        
        labels = df_merged["definition"]
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 提取指定层的隐藏状态
        num_layers = hidden_states.shape[1]
        selected_layers = [l for l in layers if l < num_layers]
        
        if not selected_layers:
            logger.error(f"指定的层 {layers} 超出了隐藏状态的层数范围 (0-{num_layers-1})。")
            return 1
        
        logger.info(f"有效的层: {selected_layers}")
        
        # 将所有选定层的隐藏状态合并在一起
        combined_hidden_states = hidden_states[:, selected_layers, :]
        # 将三维张量转换为二维：将所有选定层的特征连接在一起
        combined_hidden_states = combined_hidden_states.reshape(combined_hidden_states.shape[0], -1)
        
        logger.info(f"合并后的隐藏状态张量形状: {combined_hidden_states.shape}")
        
        save_path = os.path.join(args.output_dir, f'combined_layers_{args.method}.png')
        logger.info(f"正在绘制合并层的图表: {selected_layers}")
        
        plot_hidden_states(
            combined_hidden_states,
            labels=df_merged['definition'],
            method=args.method,
            title=f'Hidden States ({args.method}) - Layers {args.layers}',
            save_path=save_path,
            show_plot=False,
            df=df_merged,
            content_column='definition',
            top_n=3
        )
        
        # 验证文件是否成功创建
        if os.path.exists(save_path):
            logger.info(f"图表已成功保存到: {save_path}")
        else:
            logger.error(f"图表可能未成功保存，文件不存在: {save_path}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"发生未捕获的异常: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)