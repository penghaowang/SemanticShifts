import argparse
import torch
import os
import numpy as np
import logging
import pandas as pd

# Import functions written earlier from evaluate.py
from evaluate import merge_label, resolve_file_path, get_time_periods
from plot import plot_hidden_states_layers, fix_tensor_from_series
from logger_config import setup_logger

# Set up logging
logger = setup_logger('run_plot', 'logs/run_plot.log')

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Run plot_hidden_states_layers with hidden states and optional labels")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to hidden states tensor file (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=True, help='Path to the sentence CSV file')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to the label CSV file')
    parser.add_argument('--output_dir', type=str, default='plot_output', help='Where to save the plots')
    parser.add_argument('--method', type=str, default='umap', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='Dimensionality reduction method')
    parser.add_argument('--layers', type=str, required=False, help='Layers to analyze, e.g., "0,1,2" or "0-5"')
    parser.add_argument('--word', type=str, required=False, help='Target word')
    parser.add_argument('--extraction_method', type=str, required=False, help='Extraction method')
    args = parser.parse_args()

    # Find hidden states file
    hidden_states_file = resolve_file_path(
        args.hidden_states_dir,
        "*.pt", 
        f"Hidden states file not found: {args.hidden_states_dir}",
        logger
    )
    
    if hidden_states_file is None:
        return

    # Check if necessary CSV files exist
    if not os.path.exists(args.sentence_csv):
        logger.error(f"Sentence CSV file does not exist: {args.sentence_csv}")
        return
    
    if not os.path.exists(args.label_csv):
        logger.error(f"Label CSV file does not exist: {args.label_csv}")
        return

    # Execute merge_label
    logger.info("Starting merge and plotting labeled scatter plot.")
    df_merged = merge_label(
        hs_pt_file=hidden_states_file,
        sentence_csv_file=args.sentence_csv,
        label_csv_file=args.label_csv
    )
    # filter out -1 label
    df_merged = df_merged[df_merged['label_index'] != -1]
    if df_merged.empty:
        logger.error("Merge result is empty, cannot plot.")
        return

    logger.info(f"Merge completed, DataFrame rows: {len(df_merged)}")
    
    # Get predefined time periods
    time_periods = get_time_periods()
    logger.info(f"Using predefined time periods: {time_periods}")
    
    # Generate plots for each predefined time period
    for period in time_periods:
        # Parse time period string, e.g., "1990-2000"
        start_year, end_year = map(int, period.split('-'))
        
        # Filter data by year
        period_df = df_merged[(df_merged['year'] >= start_year) & (df_merged['year'] <= end_year)]
        
        if period_df.empty:
            continue
        
        period_hidden_states = fix_tensor_from_series(period_df["hidden_states"])
        period_labels = period_df["definition"]
        
        # Create separate output directory for each time period
        period_dir = os.path.join(args.output_dir, f"period_{period}")
        os.makedirs(period_dir, exist_ok=True)
        
        # Plot for this time period
        plot_hidden_states_layers(
            period_hidden_states, 
            labels=period_labels, 
            method=args.method, 
            save_dir=period_dir, 
            show_plots=False
        )
    
    logger.info(f"Plots for all time periods saved to directory: {args.output_dir}")

if __name__ == "__main__":
    main()