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
    # Required arguments
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                      help='List of dataset paths')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Model name or path')
    parser.add_argument('--target_words', type=str, required=True,
                      help='Target words and POS tags, format "word,POS", separated by space')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--max_length', type=int, default=2048,
                      help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker processes for data loading')
    parser.add_argument('--output_dir', type=str, default="datasets",
                      help='Output directory')
    parser.add_argument('--context_mode', type=str, default="sentence",
                      choices=['sentence', 'token'],
                      help='Context mode: sentence or token')
    parser.add_argument('--context_window', type=int, default=3,
                      help='Context window size')
    parser.add_argument('--duplicate_handling', type=str, default="mask",
                      choices=['mask', 'remove'],
                      help='Duplicate handling method')
    parser.add_argument('--test_batches', type=int, default=None,
                      help='Number of batches for testing, None means process all data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('run_dataloader.log')
        ]
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse target words and POS tags
    target_words = []
    for word_pos in args.target_words.split():
        word, pos = word_pos.split(',')
        target_words.append((word, pos))
    
    logging.info(f"Target words: {target_words}")
    
    # Load tokenizer
    logging.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load datasets
    dfs = []
    for path in args.dataset_paths:
        logging.info(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        logging.info(f"Column names: {df.columns.tolist()}")
        logging.info(f"Loaded {len(df)} records")
        dfs.append(df)
    
    # Concatenate datasets
    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total records after concatenation: {len(df)}")
    
    # Initialize data loader
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
    
    # Load and process datasets
    try:
        logging.info(f"Starting dataset processing")
        
        # Process each target word separately
        for word, pos in target_words:
            word_pos_dir = f"{word}_{pos}"
            logging.info(f"Processing {word}:{pos}")
            
            # Set output directory for the word
            word_output_dir = os.path.join(
                args.output_dir,
                f"context_{args.context_window}_{args.context_mode}",
                word_pos_dir
            )
            os.makedirs(word_output_dir, exist_ok=True)
            
            # Process dataset for the word
            dataset = dataloader.load_dataset(
                data_paths=args.dataset_paths,
                target_word=(word, pos),
                split_ratio=0  # Do not split for test set
            )
            
            # Save dataset for the word
            dataset.save_to_disk(word_output_dir)
            logging.info(f"Dataset saved to: {word_output_dir}")
            
            # Print dataset statistics
            if isinstance(dataset, dict):
                for split, ds in dataset.items():
                    logging.info(f"{split} set size: {len(ds)}")
                    logging.info(f"Dataset features: {ds.features}")
            else:
                logging.info(f"Dataset size: {len(dataset)}")
                logging.info(f"Dataset features: {dataset.features}")
            
    except Exception as e:
        logging.error(f"Error during dataset processing: {str(e)}")
        raise

if __name__ == '__main__':
    main()