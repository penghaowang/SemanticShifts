# run_preprocessing.py

import logging
from preprocessing import TextPreprocessor
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import pandas as pd

def main(args):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("run_preprocessing.log"),
            logging.StreamHandler()
        ]
    )
    # Check GPU availability
    logging.info(f"Checking GPU availability")
    if torch.cuda.is_available():
        logging.info(f"GPU is available. Total CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.warning("No CUDA-capable GPU found. Running on CPU.")
    logging.info("Starting Preprocessing Process")
    
    # Initialize accelerator with mixed precision if specified
    mixed_precision = "fp16" if args.mixed_precision else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    logging.info(f"Using device: {accelerator.device}")
    logging.info(f"Mixed precision: {mixed_precision}")
    
    # Load model and tokenizer for perplexity calculation
    if args.calc_perplexity:
        logging.info(f"Loading model for perplexity: {args.perplexity_model}")
        logging.info(f"Using 8-bit inference: {args.use_8bit}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.perplexity_model)
        
        # Load model with quantization if specified
        if args.use_gptq:
            from auto_gptq import AutoGPTQForCausalLM
            model = AutoGPTQForCausalLM.from_quantized(
                args.perplexity_model,
                device_map="auto",
                use_triton=False,
                quantize_config={
                    "bits": args.gptq_bits,
                    "group_size": args.gptq_group_size,
                }
            )
        elif args.use_8bit:
            logging.warning("8-bit quantization is deprecated, please use GPTQ instead")
            model = AutoModelForCausalLM.from_pretrained(
                args.perplexity_model,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.perplexity_model,
                device_map="auto"
            )
            model = accelerator.prepare(model)
            logging.info(f"Model loaded successfully")
    else:
        model = None
        tokenizer = None
    
    # Initialize TextPreprocessor with the configured accelerator
    preprocessor = TextPreprocessor(
        accelerator=accelerator,
        spacy_model=args.spacy_model,
        max_batch_size=args.max_batch_size,
        num_workers=args.num_workers
    )
    
    # Load dataset
    logging.info(f"Loading dataset from {args.dataset_path}")
    df = preprocessor.load_dataset(args.dataset_path)
    
    # Create a copy before modifications
    df = df.copy()
    
    # Fix text errors
    logging.info("Fixing text errors in sentences")
    df['sentence'] = df['sentence'].apply(preprocessor.fix_text_errors)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataframe
    logging.info("Processing dataframe")
    try:
        processed_df = preprocessor.process_dataframe(
            df,
            model=model,
            tokenizer=tokenizer,
            calc_perplexity=args.calc_perplexity
        )
        
        # Save processed DataFrame to CSV
        model_name = args.perplexity_model.split('/')[-1] if args.calc_perplexity else "no_perplexity"
        precision_info = "_8bit" if args.use_8bit else "_fp16" if args.mixed_precision else ""
        input_filename = os.path.splitext(os.path.basename(args.dataset_path))[0]
        output_path = os.path.join(args.output_dir, f"{input_filename}_{model_name}{precision_info}.csv")
        preprocessor.save_to_csv(processed_df, output_path)
        logging.info(f"Processed data saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise
    
    logging.info("Preprocessing Process Completed Successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Preprocessing")
    
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the processed dataset (from run_data_loader)')
    parser.add_argument('--output_dir', type=str, default="processed_data",
                        help='Directory to save the preprocessed CSV')
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm',
                        help='SpaCy model to use for sentence analysis')
    parser.add_argument('--max_batch_size', type=int, default=512,
                        help='Maximum batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel processing')
    parser.add_argument('--calc_perplexity', action='store_true',
                        help='Whether to calculate perplexity')
    parser.add_argument('--perplexity_model', type=str, default='gpt2',
                        help='Model to use for perplexity calculation')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision (fp16) training')
    parser.add_argument('--use_8bit', action='store_true',
                        help='Use 8-bit quantization for the model')
    parser.add_argument('--use_gptq', action='store_true',
                        help='Use GPTQ quantization for the model')
    parser.add_argument('--gptq_bits', type=int, default=4,
                        help='Number of bits for GPTQ quantization (default: 4)')
    parser.add_argument('--gptq_group_size', type=int, default=128, 
                        help='Group size for GPTQ quantization (default: 128)')
    
    args = parser.parse_args()
    
    main(args)
# python run_preprocessing.py --dataset_path data/cs_bulletin_pdf_en_1014.csv --output_dir processed_data --spacy_model en_core_web_sm --max_batch_size 64 --num_workers 4 --calc_perplexity --perplexity_model meta-llama/Llama-3.1-8B --mixed_precision --use_8bit