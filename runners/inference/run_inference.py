import argparse
import os
import pandas as pd
from typing import List, Optional, Dict
from inference import ModelInference
from prompt_templates import PROMPT_TEMPLATES
from dataloader import CustomDataLoader
from transformers import AutoTokenizer
import traceback
import sys

# 导入自定义日志配置
from logger_config import setup_logger

# 设置模块特定的日志记录器
logger = setup_logger('run_inference', 'logs/run_inference.log')

def parse_args():
    parser = argparse.ArgumentParser()
    # 数据相关参数
    parser.add_argument('--data_path', type=str, required=False,
                      help='原始数据路径，如果使用预处理数据集则不需要')
    parser.add_argument('--saved_dataset_dir', type=str, required=False,
                      help='预处理数据集目录路径')
    
    # 模型相关参数
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=2048)
    
    # 提取方法相关参数
    parser.add_argument('--layer_indices', type=str, default='all')
    parser.add_argument('--prompt_template', type=str, default='basic',
                      choices=list(PROMPT_TEMPLATES.keys()))
    parser.add_argument('--extraction_method', type=str, default='input_last_token',
                      choices=['input_last_token', 'eos_token', 'input_mean', 'output_mean', 'output_eos', 'all'])
    parser.add_argument('--use_all_layers', action='store_true')
    
    # 目标词相关参数
    parser.add_argument('--target_words', type=str, required=True,
                      help='Target word and POS tag in format "word:POS"')
    
    # 数据处理相关参数（仅在使用原始数据时需要）
    parser.add_argument('--context_mode', type=str, default='sentence',
                      choices=['sentence', 'window', 'document'])
    parser.add_argument('--context_window', type=int, default=3,
                      help='上下文窗口大小（仅在context_mode为window时有效）')
    parser.add_argument('--min_sentence_length', type=int, default=5,
                      help='最小句子长度')
    parser.add_argument('--max_sentence_length', type=int, default=100,
                      help='最大句子长度')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='最大样本数量')
    parser.add_argument('--duplicate_handling', type=str, default="mask",
                      choices=['mask', 'remove'],
                      help='重复处理方式：mask或remove')
    
    # 日志相关参数
    parser.add_argument('--log_file', type=str, default='logs/run_inference.log',
                      help='日志文件路径')
    
    # 添加8位量化参数
    parser.add_argument('--use_8bit', action='store_true',
                      help='使用8位量化进行推理，可减少内存占用')
    
    # 添加保存隐藏状态参数
    parser.add_argument('--save_hidden_states', action='store_true',
                      help='保存合并的隐藏状态')
    
    args = parser.parse_args()
    
    # 处理目标词参数
    if args.target_words:
        parts = args.target_words.split(':')
        if len(parts) == 2:
            args.target_word, args.target_pos = parts
        else:
            args.target_word = args.target_words
            args.target_pos = None
    
    # 处理层索引参数
    if args.layer_indices != 'all':
        try:
            if ',' in args.layer_indices:
                args.layer_indices = [int(idx) for idx in args.layer_indices.split(',')]
            elif '-' in args.layer_indices:
                start, end = map(int, args.layer_indices.split('-'))
                args.layer_indices = list(range(start, end + 1))
            else:
                args.layer_indices = [int(args.layer_indices)]
            # 当指定了具体的层索引时，use_all_layers应该为False
            args.use_all_layers = False
        except:
            logger.error(f"无效的层索引格式: {args.layer_indices}")
            args.layer_indices = None
            # 当层索引格式无效时，使用所有层
            args.use_all_layers = True
    else:
        args.layer_indices = None  # 使用所有层
        args.use_all_layers = True  # 当指定'all'时，use_all_layers应该为True
    
    return args

def main():
    args = parse_args()
    
    # 如果指定了自定义日志文件，重新配置日志记录器
    if args.log_file != 'logs/run_inference.log':
        global logger
        logger = setup_logger('run_inference', args.log_file)
    
    try:
        # 将目标词中的冒号替换为下划线以创建安全的目录名
        safe_output_dir = args.output_dir
        if args.target_words:
            for target_word in args.target_words:
                if ':' in target_word:
                    safe_output_dir = safe_output_dir.replace(target_word, target_word.replace(':', '_'))
        
        # 创建输出目录
        os.makedirs(safe_output_dir, exist_ok=True)
        
        # 记录关键配置参数
        logger.info("=== 运行配置 ===")
        logger.info(f"目标词: {args.target_words}")
        logger.info(f"提取方法: {args.extraction_method}")
        logger.info(f"模型: {args.model_name}")
        logger.info(f"输出目录: {safe_output_dir}")
        logger.info("=============")
        
        # 初始化模型推理
        model_inference = ModelInference(
            model_name=args.model_name,
            layer_indices=args.layer_indices,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            prompt_template=PROMPT_TEMPLATES[args.prompt_template],
            extraction_method=args.extraction_method,
            use_all_layers=args.use_all_layers,
            target_word=args.target_word,
            target_pos=args.target_pos,
            saved_dataset_dir=args.saved_dataset_dir,
            output_dir=safe_output_dir,
            save_hidden_states=args.save_hidden_states,
            log_file=args.log_file,
            use_8bit=args.use_8bit
        )
        
        # 如果使用原始数据，需要先处理
        if args.data_path:
            logger.info("使用原始数据进行处理...")
            # 初始化tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
            # 初始化CustomDataLoader
            dataloader = CustomDataLoader(
                tokenizer=tokenizer,
                target_words=[(args.target_word, args.target_pos)],
                batch_size=args.batch_size,
                max_length=args.max_length,
                context_mode=args.context_mode,
                context_window=args.context_window,
                min_sentence_length=args.min_sentence_length,
                max_sentence_length=args.max_sentence_length,
                max_samples=args.max_samples,
                duplicate_handling=args.duplicate_handling
            )
            
            # 加载并处理数据
            df = dataloader.load_and_process_data(args.data_path)
            if df is None or len(df) == 0:
                logger.error("数据处理失败或没有符合条件的样本")
                return 1
            
            logger.info(f"处理后的数据集大小: {len(df)}")
        else:
            logger.info("使用预处理数据集...")
            df = None
        
        # 运行推理
        logger.info("开始运行推理...")
        results = model_inference.run_inference(df)
        
        # 保存结果
        experiment_name = f"{args.prompt_template}_{args.extraction_method}_{args.target_word}_{args.target_pos}"
        save_path = model_inference.save_results(results, experiment_name)
        logger.info(f'结果已保存到: {save_path}')
        
    except Exception as e:
        logger.error(f"处理过程发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
