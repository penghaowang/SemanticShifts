import argparse
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from logger_config import setup_logger
from dataloader import CustomDataLoader

logger = setup_logger('create_data', 'logs/create_data.log')

# Define the target word list
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
    """Create a list of target words and their POS tags"""
    word_pairs = []
    # Add adjectives
    for adj in TARGET_ADJECTIVES:
        word_pairs.append((adj, 'ADJ'))
    # Add nouns
    for noun in TARGET_NOUNS:
        word_pairs.append((noun, 'NOUN'))
    return word_pairs

def get_dataset_path(output_dir: str, word: str, pos: str) -> Path:
    """Generate the save or load path for the dataset"""
    return Path(output_dir) / f"{word}_{pos}"

def save_dataset(dataset: Dataset, output_dir: str, word: str, pos: str) -> None:
    """Save the dataset to the specified directory"""
    save_path = get_dataset_path(output_dir, word, pos)
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_path)
    logger.info(f"Saved {word}:{pos} dataset to {save_path}")

def load_dataset(output_dir: str, word: str, pos: str) -> Optional[Dataset]:
    """Load the dataset from the saved directory"""
    load_path = get_dataset_path(output_dir, word, pos)
    if load_path.exists():
        try:
            dataset = Dataset.load_from_disk(load_path)
            logger.info(f"Loaded {word}:{pos} dataset from {load_path}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load {word}:{pos} dataset: {e}")
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
    """Process data and save/load datasets"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get the list of target word pairs
    target_word_pairs = create_target_word_pairs()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Store all datasets
    datasets = {}
    
    for word, pos in target_word_pairs:
        logger.info(f"Processing {word}:{pos}")
        
        # Check if the saved dataset already exists
        if not force_reload:
            saved_dataset = load_dataset(output_dir, word, pos)
            if saved_dataset is not None:
                datasets[f"{word}_{pos}"] = saved_dataset
                continue
        
        # Initialize data loader
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
            # Load and process data
            dataset = dataloader.load_dataset(
                data_paths=data_paths,
                split_ratio=0  # Do not split the dataset
            )
            
            # Save the dataset
            save_dataset(dataset, output_dir, word, pos)
            
            # Add to the results dictionary
            datasets[f"{word}_{pos}"] = dataset
            
        except Exception as e:
            logger.error(f"Failed to process {word}:{pos}: {e}")
            continue
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Create and save target word datasets")
    parser.add_argument('--data_paths', type=str, nargs='+', required=True,
                      help='Input data file paths')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Model name')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--max_length', type=int, default=2048,
                      help='Maximum sequence length')
    parser.add_argument('--context_window', type=int, default=3,
                      help='Context window size')
    parser.add_argument('--force_reload', action='store_true',
                      help='Force reload all datasets')
    
    args = parser.parse_args()
    
    # Process data
    datasets = process_data(
        data_paths=args.data_paths,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        context_window=args.context_window,
        force_reload=args.force_reload
    )
    
    # Print statistics
    logger.info("Dataset statistics:")
    for name, dataset in datasets.items():
        logger.info(f"{name}: {len(dataset)} samples")

if __name__ == '__main__':
    main()