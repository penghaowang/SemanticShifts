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

# Import custom logger configuration
from logger_config import setup_logger

# Set up module-specific logger
logger = setup_logger('run_inference', 'logs/run_inference.log')

def parse_args():
    parser = argparse.ArgumentParser()
    # Data-related arguments
    parser.add_argument('--data_path', type=str, required=False,
                      help='Path to raw data, not needed if using preprocessed dataset')
    parser.add_argument('--saved_dataset_dir', type=str, required=False,
                      help='Directory path for preprocessed dataset')
    
    # Model-related arguments
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=2048)
    
    # Extraction method arguments
    parser.add_argument('--layer_indices', type=str, default='all')
    parser.add_argument('--prompt_template', type=str, default='basic',
                      choices=list(PROMPT_TEMPLATES.keys()))
    parser.add_argument('--extraction_method', type=str, default='input_last_token',
                      choices=['input_last_token', 'eos_token', 'input_mean', 'output_mean', 'output_eos', 'all'])
    parser.add_argument('--use_all_layers', action='store_true')
    
    # Target word arguments
    parser.add_argument('--target_words', type=str, required=True,
                      help='Target word and POS tag in format "word:POS"')
    
    # Data processing arguments (only needed when using raw data)
    parser.add_argument('--context_mode', type=str, default='sentence',
                      choices=['sentence', 'window', 'document'])
    parser.add_argument('--context_window', type=int, default=3,
                      help='Context window size (only effective when context_mode is window)')
    parser.add_argument('--min_sentence_length', type=int, default=5,
                      help='Minimum sentence length')
    parser.add_argument('--max_sentence_length', type=int, default=100,
                      help='Maximum sentence length')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples')
    parser.add_argument('--duplicate_handling', type=str, default="mask",
                      choices=['mask', 'remove'],
                      help='Duplicate handling method: mask or remove')
    
    # Logging arguments
    parser.add_argument('--log_file', type=str, default='logs/run_inference.log',
                      help='Log file path')
    
    # Add 8-bit quantization argument
    parser.add_argument('--use_8bit', action='store_true',
                      help='Use 8-bit quantization for inference to reduce memory usage')
    
    # Add save hidden states argument
    parser.add_argument('--save_hidden_states', action='store_true',
                      help='Save combined hidden states')
    
    args = parser.parse_args()
    
    # Process target word argument
    if args.target_words:
        parts = args.target_words.split(':')
        if len(parts) == 2:
            args.target_word, args.target_pos = parts
        else:
            args.target_word = args.target_words
            args.target_pos = None
    
    # Process layer indices argument
    if args.layer_indices != 'all':
        try:
            if ',' in args.layer_indices:
                args.layer_indices = [int(idx) for idx in args.layer_indices.split(',')]
            elif '-' in args.layer_indices:
                start, end = map(int, args.layer_indices.split('-'))
                args.layer_indices = list(range(start, end + 1))
            else:
                args.layer_indices = [int(args.layer_indices)]
            # When specific layer indices are given, use_all_layers should be False
            args.use_all_layers = False
        except:
            logger.error(f"Invalid layer index format: {args.layer_indices}")
            args.layer_indices = None
            # When layer index format is invalid, use all layers
            args.use_all_layers = True
    else:
        args.layer_indices = None  # Use all layers
        args.use_all_layers = True  # When 'all' is specified, use_all_layers should be True
    
    return args

def main():
    args = parse_args()
    
    # If a custom log file is specified, reconfigure the logger
    if args.log_file != 'logs/run_inference.log':
        global logger
        logger = setup_logger('run_inference', args.log_file)
    
    try:
        # Replace colon in target words with underscore for safe directory name
        safe_output_dir = args.output_dir
        if args.target_words:
            for target_word in args.target_words:
                if ':' in target_word:
                    safe_output_dir = safe_output_dir.replace(target_word, target_word.replace(':', '_'))
        
        # Create output directory
        os.makedirs(safe_output_dir, exist_ok=True)
        
        # Log key configuration parameters
        logger.info("=== Run Configuration ===")
        logger.info(f"Target words: {args.target_words}")
        logger.info(f"Extraction method: {args.extraction_method}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Output directory: {safe_output_dir}")
        logger.info("=============")
        
        # Initialize ModelInference
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
        
        # If using raw data, process it first
        if args.data_path:
            logger.info("Processing using raw data...")
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
            # Initialize CustomDataLoader
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
            
            # Load and process data
            df = dataloader.load_and_process_data(args.data_path)
            if df is None or len(df) == 0:
                logger.error("Data processing failed or no eligible samples found")
                return 1
            
            logger.info(f"Size of processed dataset: {len(df)}")
        else:
            logger.info("Using preprocessed dataset...")
            df = None
        
        # Run inference
        logger.info("Starting inference run...")
        results = model_inference.run_inference(df)
        
        # Save results
        experiment_name = f"{args.prompt_template}_{args.extraction_method}_{args.target_word}_{args.target_pos}"
        save_path = model_inference.save_results(results, experiment_name)
        logger.info(f'Results saved to: {save_path}')
        
    except Exception as e:
        logger.error(f"Error occurred during processing: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
