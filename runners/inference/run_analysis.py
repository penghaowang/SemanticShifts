#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run hidden states analysis and visualization.
This script finds all hidden states files in a directory and analyzes them.
"""

import os
import argparse
import glob
from pathlib import Path
import subprocess
import pandas as pd
import traceback

# Import custom logger configuration
from logger_config import setup_logger, TRACE

# Set up module-specific logger
logger = setup_logger('run_analysis', 'logs/run_analysis.log')

def find_hidden_states_files(base_dir):
    """Find all hidden states files recursively in the directory"""
    # Look for common hidden states file patterns
    patterns = [
        "**/*states.pt",
        "**/all_states.pt",
        "**/hidden_states/**/*.pt",
        "**/sample_*.pt"
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    
    # Remove duplicates and sort
    unique_files = sorted(set(all_files))
    logger.info(f"Found {len(unique_files)} hidden state files")
    
    return unique_files

def find_word_data(base_dir):
    """Find corresponding word data CSV files"""
    # Look for metadata or generations CSV files
    patterns = [
        "**/generations.csv",
        "**/metadata.csv",
        "**/*_info.csv"
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    
    # Return the first file found, or None
    if all_files:
        logger.info(f"Found word data file: {all_files[0]}")
        return all_files[0]
    else:
        logger.warning("Word data file not found")
        return None

def run_analysis(hidden_states_file, word_data_file, output_dir, reduction_method):
    """Run analysis on a single hidden states file"""
    # Create command
    cmd = [
        "python", "plot.py",
        "--input_path", hidden_states_file,
        "--output_dir", output_dir,
        "--reduction_method", reduction_method
    ]
    
    # Add word data if available
    if word_data_file:
        cmd.extend(["--word_data_path", word_data_file])
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Completed analysis for {hidden_states_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error analyzing {hidden_states_file}: {e}")
        logger.error(f"Standard output: {e.stdout}")
        logger.error(f"Standard error: {e.stderr}")
        logger.debug(traceback.format_exc())
        return False

def main(args):
    """Main function to analyze all hidden states files"""
    # If a custom log file is specified, reconfigure the logger
    if args.log_file != 'logs/run_analysis.log':
        global logger
        logger = setup_logger('run_analysis', args.log_file)
    
    # Find all hidden states files
    hidden_states_files = find_hidden_states_files(args.input_dir)
    
    if not hidden_states_files:
        logger.error(f"No hidden state files found in {args.input_dir}")
        return
    
    # Find word data file
    word_data_file = args.word_data_path or find_word_data(args.input_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    successful = 0
    for i, hidden_states_file in enumerate(hidden_states_files):
        logger.info(f"Processing file {i+1}/{len(hidden_states_files)}: {hidden_states_file}")
        
        # Create subdirectory based on file name
        file_base_name = Path(hidden_states_file).stem
        method_name = Path(hidden_states_file).parent.name
        target_word = Path(hidden_states_file).parent.parent.name if "hidden_states" in str(Path(hidden_states_file).parent) else "unknown"
        
        # Create a descriptive output directory
        file_output_dir = os.path.join(
            args.output_dir,
            f"{target_word}_{method_name}_{file_base_name}"
        )
        
        # Run analysis
        success = run_analysis(
            hidden_states_file,
            word_data_file,
            file_output_dir,
            args.reduction_method
        )
        
        if success:
            successful += 1
    
    logger.info(f"Analysis completed. Successfully processed {successful}/{len(hidden_states_files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch analyze hidden state files")
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing hidden state files')
    parser.add_argument('--word_data_path', type=str, default=None,
                        help='Path to CSV file containing word information (optional)')
    parser.add_argument('--output_dir', type=str, default="analysis_results",
                        help='Directory to save analysis output')
    parser.add_argument('--reduction_method', type=str, default='pca',
                        choices=['pca', 'tsne'],
                        help='Dimensionality reduction method for visualization')
    parser.add_argument('--log_file', type=str, default='logs/run_analysis.log',
                        help='Log file path')
    
    args = parser.parse_args()
    main(args)
