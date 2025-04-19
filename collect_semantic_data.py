#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收集所有词语的语义变化数据，包括JSD和熵值
从已生成的CSV文件中读取，合并为汇总CSV文件
"""

import os
import pandas as pd
import glob
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any
from logger_config import setup_logger

# 设置日志
logger = setup_logger('collect_semantic_data', 'logs/collect_semantic_data.log')

def load_and_combine_csv_files(file_pattern: str, description: str = "数据") -> Optional[pd.DataFrame]:
    """
    加载并合并符合指定模式的所有CSV文件
    
    参数:
        file_pattern: 文件路径匹配模式，使用glob格式
        description: 数据类型描述，用于日志记录
        
    返回:
        合并的DataFrame，如果没有文件则返回None
    """
    files = glob.glob(file_pattern)
    if not files:
        logger.warning(f"未找到匹配 {file_pattern} 的CSV文件")
        return None
    
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            logger.info(f"已读取 {file}")
        except Exception as e:
            logger.error(f"读取 {file} 时出错: {str(e)}")
    
    if not dataframes:
        logger.error(f"未找到任何有效的{description}文件")
        return None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"已合并 {len(combined_df)} 行{description}")
    
    return combined_df

def process_word_directory(args: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """
    处理单个词语目录，加载其JSD和熵值数据
    
    参数:
        args: 包含(word_dir, base_dir, time_bin_level, full_path)的元组
        
    返回:
        包含该词语所有数据的字典
    """
    word_dir, base_dir, time_bin_level, full_path = args
    result = {"word": word_dir}
    
    # 查找该词语的JSD CSV文件
    jsd_pattern = os.path.join(full_path, f"jsd_changes_{time_bin_level}_*.csv")
    df_jsd = load_and_combine_csv_files(jsd_pattern, f"词语 '{word_dir}' 的JSD数据")
    result["jsd"] = df_jsd
    
    # 查找该词语的熵值CSV文件
    entropy_pattern = os.path.join(full_path, f"entropy_changes_{time_bin_level}_*.csv")
    df_entropy = load_and_combine_csv_files(entropy_pattern, f"词语 '{word_dir}' 的熵值数据")
    result["entropy"] = df_entropy
    
    # 查找该词语的多义词意思熵值CSV文件
    meaning_pattern = os.path.join(full_path, f"meaning_entropy_changes_{time_bin_level}_*.csv")
    df_meaning = load_and_combine_csv_files(meaning_pattern, f"词语 '{word_dir}' 的多义词意思熵值数据")
    result["meaning_entropy"] = df_meaning
    
    return result

def collect_all_data(base_dir: str = 'semantic_shift_plots', time_bin_level: str = 'period') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    并行收集所有词语的JSD和熵值数据
    
    参数:
        base_dir: 包含所有词语目录的基础目录
        time_bin_level: 时间分组级别 ('period' 或 'year')
    
    返回:
        (combined_jsd, combined_entropy, combined_meaning_entropy)元组
    """
    # 获取所有词语目录
    word_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    logger.info(f"发现 {len(word_dirs)} 个词语目录")
    
    all_jsd_data = []
    all_entropy_data = []
    all_meaning_entropy_data = []
    
    # 准备并行处理的参数
    process_args = [(word_dir, base_dir, time_bin_level, os.path.join(base_dir, word_dir)) for word_dir in word_dirs]
    
    # 并行处理所有词语目录
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_word_directory, process_args))
    
    # 整理结果
    for result in results:
        if result["jsd"] is not None:
            all_jsd_data.append(result["jsd"])
        if result["entropy"] is not None:
            all_entropy_data.append(result["entropy"])
        if result["meaning_entropy"] is not None:
            all_meaning_entropy_data.append(result["meaning_entropy"])
    
    # 合并所有词语的JSD数据
    combined_jsd = None
    if all_jsd_data:
        combined_jsd = pd.concat(all_jsd_data, ignore_index=True)
        logger.info(f"总共合并了 {len(combined_jsd)} 行JSD数据")
    else:
        logger.error("未找到任何JSD数据文件")
    
    # 合并所有词语的熵值数据
    combined_entropy = None
    if all_entropy_data:
        combined_entropy = pd.concat(all_entropy_data, ignore_index=True)
        logger.info(f"总共合并了 {len(combined_entropy)} 行熵值数据")
    else:
        logger.error("未找到任何熵值数据文件")
    
    # 合并所有多义词的意思熵值数据
    combined_meaning_entropy = None
    if all_meaning_entropy_data:
        combined_meaning_entropy = pd.concat(all_meaning_entropy_data, ignore_index=True)
        logger.info(f"总共合并了 {len(combined_meaning_entropy)} 行多义词意思熵值数据")
    
    return combined_jsd, combined_entropy, combined_meaning_entropy

def save_data_and_create_pivot(df: pd.DataFrame, output_dir: str, file_prefix: str, 
                              time_bin_level: str, pivot_params: Dict[str, Any]) -> None:
    """
    保存数据并创建交叉表
    
    参数:
        df: 要保存的数据框
        output_dir: 输出目录
        file_prefix: 文件名前缀
        time_bin_level: 时间分组级别
        pivot_params: 创建交叉表的参数
    """
    # 保存合并后的数据
    output_file = os.path.join(output_dir, f'{file_prefix}_{time_bin_level}.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"已将数据保存至 {output_file}")
    
    # 创建交叉表
    try:
        pivot_df = pd.pivot_table(df, **pivot_params)
        pivot_output = os.path.join(output_dir, f'{file_prefix}_by_word_time_{time_bin_level}.csv')
        pivot_df.to_csv(pivot_output)
        logger.info(f"已创建交叉表，保存至 {pivot_output}")
    except Exception as e:
        logger.error(f"创建交叉表时出错: {str(e)}")

def main():
    """主函数"""
    base_dir = 'semantic_shift_plots'
    time_bin_level = 'period'  # 可选 'year' 或 'period'
    output_dir = 'semantic_shift_summary'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 并行收集所有数据
    combined_jsd, combined_entropy, combined_meaning_entropy = collect_all_data(base_dir, time_bin_level)
    
    # 处理JSD数据
    if combined_jsd is not None:
        save_data_and_create_pivot(
            combined_jsd, 
            output_dir, 
            'all_words_jsd',
            time_bin_level,
            {
                'values': 'js_div',
                'index': ['word', 'pos'],
                'columns': 'time_pair',
                'aggfunc': 'mean'
            }
        )
    
    # 处理熵值数据
    if combined_entropy is not None:
        save_data_and_create_pivot(
            combined_entropy, 
            output_dir, 
            'all_words_entropy',
            time_bin_level,
            {
                'values': 'avg_entropy',
                'index': ['word', 'pos'],
                'columns': 'time_bin',
                'aggfunc': 'mean'
            }
        )
    
    # 处理多义词意思熵值数据
    if combined_meaning_entropy is not None:
        output_file = os.path.join(output_dir, f'all_words_meaning_entropy_{time_bin_level}.csv')
        combined_meaning_entropy.to_csv(output_file, index=False)
        logger.info(f"已将所有多义词的意思熵值数据保存至 {output_file}")

if __name__ == "__main__":
    main()