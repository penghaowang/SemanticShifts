import argparse
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from logger_config import setup_logger
from dataloader import CustomDataLoader

logger = setup_logger('create_data', 'logs/create_data.log')

# 定义目标词列表
TARGET_ADJECTIVES = [
    'industrial', 'traditional', 'monetary', 'inflationary', 'foreign',
    'public', 'private', 'corporate', 'real', 'financial',
    'available', 'strong', 'stable', 'fair', 'competitive',
    'annual', 'financial', 'net'
]

TARGET_NOUNS = [
    'market', 'rate', 'bank', 'interest', 'investment', 'bond',
    'share', 'capital', 'exchange', 'tax', 'growth', 'security',
    'company', 'dollar', 'debt', 'equity', 'profit', 'loss',
    'gain', 'decline', 'import', 'export'
]

def create_target_word_pairs() -> List[Tuple[str, str]]:
    """创建目标词和词性对的列表"""
    word_pairs = []
    # 添加形容词
    for adj in TARGET_ADJECTIVES:
        word_pairs.append((adj, 'ADJ'))
    # 添加名词
    for noun in TARGET_NOUNS:
        word_pairs.append((noun, 'NOUN'))
    return word_pairs

def get_dataset_path(output_dir: str, word: str, pos: str) -> Path:
    """生成数据集保存或加载路径"""
    return Path(output_dir) / f"{word}_{pos}"

def save_dataset(dataset: Dataset, output_dir: str, word: str, pos: str) -> None:
    """保存数据集到指定目录"""
    save_path = get_dataset_path(output_dir, word, pos)
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_path)
    logger.info(f"已保存 {word}:{pos} 数据集到 {save_path}")

def load_dataset(output_dir: str, word: str, pos: str) -> Optional[Dataset]:
    """从保存的目录加载数据集"""
    load_path = get_dataset_path(output_dir, word, pos)
    if load_path.exists():
        try:
            dataset = Dataset.load_from_disk(load_path)
            logger.info(f"已加载 {word}:{pos} 数据集从 {load_path}")
            return dataset
        except Exception as e:
            logger.error(f"加载 {word}:{pos} 数据集失败: {e}")
    return None

def process_data(
    data_paths: List[str],
    model_name: str,
    output_dir: str,
    batch_size: int = 32,
    max_length: int = 2048,
    context_window: int = 3,
    force_reload: bool = False
) -> Dict[str, Dataset]:
    """处理数据并保存/加载数据集"""
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 获取目标词对列表
    target_word_pairs = create_target_word_pairs()
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 存储所有数据集
    datasets = {}
    
    for word, pos in target_word_pairs:
        logger.info(f"处理 {word}:{pos}")
        
        # 检查是否已存在保存的数据集
        if not force_reload:
            saved_dataset = load_dataset(output_dir, word, pos)
            if saved_dataset is not None:
                datasets[f"{word}_{pos}"] = saved_dataset
                continue
        
        # 初始化数据加载器
        dataloader = CustomDataLoader(
            tokenizer=tokenizer,
            target_words=[(word, pos)],
            batch_size=batch_size,
            max_length=max_length,
            context_mode="sentence",
            context_window=context_window,
            duplicate_handling="remove"
        )
        
        try:
            # 加载和处理数据
            dataset = dataloader.load_dataset(
                data_paths=data_paths,
                split_ratio=0  # 不分割数据集
            )
            
            # 保存数据集
            save_dataset(dataset, output_dir, word, pos)
            
            # 添加到结果字典
            datasets[f"{word}_{pos}"] = dataset
            
        except Exception as e:
            logger.error(f"处理 {word}:{pos} 失败: {e}")
            continue
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description="创建并保存目标词数据集")
    parser.add_argument('--data_paths', type=str, nargs='+', required=True,
                      help='输入数据文件路径')
    parser.add_argument('--model_name', type=str, required=True,
                      help='模型名称')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批处理大小')
    parser.add_argument('--max_length', type=int, default=2048,
                      help='最大序列长度')
    parser.add_argument('--context_window', type=int, default=3,
                      help='上下文窗口大小')
    parser.add_argument('--force_reload', action='store_true',
                      help='强制重新加载所有数据集')
    
    args = parser.parse_args()
    
    # 处理数据
    datasets = process_data(
        data_paths=args.data_paths,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        context_window=args.context_window,
        force_reload=args.force_reload
    )
    
    # 打印统计信息
    logger.info("数据集统计:")
    for name, dataset in datasets.items():
        logger.info(f"{name}: {len(dataset)} 样本")

if __name__ == '__main__':
    main()