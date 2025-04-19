from datasets import DatasetDict
import argparse
import logging
from transformers import AutoTokenizer
from dataloader import CustomDataLoader
import random
import numpy as np
import torch
from pathlib import Path
import os
import pandas as pd

def set_seed(seed: int = 42):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                      help='数据集路径列表')
    parser.add_argument('--model_name', type=str, required=True,
                      help='模型名称或路径')
    parser.add_argument('--target_words', type=str, required=True,
                      help='目标词和词性，格式为"word,POS"，多个词用空格分隔')
    
    # 可选参数
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批处理大小')
    parser.add_argument('--max_length', type=int, default=2048,
                      help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载的工作进程数')
    parser.add_argument('--output_dir', type=str, default="datasets",
                      help='输出目录')
    parser.add_argument('--context_mode', type=str, default="sentence",
                      choices=['sentence', 'token'],
                      help='上下文模式：sentence或token')
    parser.add_argument('--context_window', type=int, default=3,
                      help='上下文窗口大小')
    parser.add_argument('--duplicate_handling', type=str, default="mask",
                      choices=['mask', 'remove'],
                      help='重复处理方式')
    parser.add_argument('--test_batches', type=int, default=None,
                      help='测试用的批次数量，None表示处理全部数据')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('run_dataloader.log')
        ]
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析目标词和词性
    target_words = []
    for word_pos in args.target_words.split():
        word, pos = word_pos.split(',')
        target_words.append((word, pos))
    
    logging.info(f"目标词: {target_words}")
    
    # 加载tokenizer
    logging.info(f"加载tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 加载数据集
    dfs = []
    for path in args.dataset_paths:
        logging.info(f"加载数据集: {path}")
        df = pd.read_csv(path)
        logging.info(f"列名: {df.columns.tolist()}")
        logging.info(f"加载了 {len(df)} 条记录")
        dfs.append(df)
    
    # 合并数据集
    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"合并后总记录数: {len(df)}")
    
    # 初始化数据加载器
    dataloader = CustomDataLoader(
        tokenizer=tokenizer,
        target_words=target_words,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        context_mode=args.context_mode,
        context_window=args.context_window,
        duplicate_handling=args.duplicate_handling
    )
    
    # 加载并处理数据集
    try:
        logging.info(f"开始处理数据集")
        
        # 为每个目标词分别处理
        for word, pos in target_words:
            word_pos_dir = f"{word}_{pos}"
            logging.info(f"Processing {word}:{pos}")
            
            # 设置该词的输出目录
            word_output_dir = os.path.join(
                args.output_dir,
                f"context_{args.context_window}_{args.context_mode}",
                word_pos_dir
            )
            os.makedirs(word_output_dir, exist_ok=True)
            
            # 处理该词的数据集
            dataset = dataloader.load_dataset(
                data_paths=args.dataset_paths,
                target_word=(word, pos),
                split_ratio=0  # 不分割测试集
            )
            
            # 保存该词的数据集
            dataset.save_to_disk(word_output_dir)
            logging.info(f"数据集已保存到: {word_output_dir}")
            
            # 打印数据集统计信息
            if isinstance(dataset, dict):
                for split, ds in dataset.items():
                    logging.info(f"{split} 集大小: {len(ds)}")
                    logging.info(f"数据集特征: {ds.features}")
            else:
                logging.info(f"数据集大小: {len(dataset)}")
                logging.info(f"数据集特征: {dataset.features}")
            
    except Exception as e:
        logging.error(f"处理数据集时出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()