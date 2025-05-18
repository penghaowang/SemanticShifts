import argparse
import torch
import os
import numpy as np
import logging

# Import functions written earlier from evaluate.py
from evaluate import merge_label, resolve_file_path
from plot import plot_hidden_states_layers, fix_tensor_from_series
from logger_config import setup_logger

# Set up logging
logger = setup_logger('run_plot', 'logs/run_plot.log')

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Run plot_hidden_states_layers with hidden states and optional labels")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to hidden states tensor file (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=False, help='Path to the sentence CSV file')
    parser.add_argument('--label_csv', type=str, required=False, help='Path to the label CSV file')
    parser.add_argument('--output_dir', type=str, default='plot_output', help='Where to save the plots')
    parser.add_argument('--method', type=str, default='zca_pca', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='Dimensionality reduction method')
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

    # If sentence_csv and label_csv are not provided, only plot hidden states (without label differentiation)
    if not args.sentence_csv or not args.label_csv:
        logger.info("sentence_csv and label_csv not provided. Will plot unlabeled scatter plot only.")
        hidden_states = torch.load(hidden_states_file)
        logger.info(f"Loaded hidden states with shape: {hidden_states.shape}")

        os.makedirs(args.output_dir, exist_ok=True)
        plot_hidden_states_layers(
            hidden_states, 
            labels=None,  # No color differentiation
            method=args.method,
            save_dir=args.output_dir, 
            show_plots=False
        )
        logger.info(f"Plots saved to directory: {args.output_dir}")
    else:
        # If sentence_csv and label_csv are provided, execute merge_label
        logger.info("sentence_csv and label_csv provided. Starting merge and plotting labeled scatter plot.")
        df_merged = merge_label(
            hs_pt_file=hidden_states_file,
            sentence_csv_file=args.sentence_csv,
            label_csv_file=args.label_csv
        )
        if df_merged.empty:
            logger.error("Merge result is empty, cannot plot.")
            return

        logger.info(f"Merge completed, DataFrame rows: {len(df_merged)}")
        # Extract hidden states and labels from df_merged, use fix_tensor_from_series to avoid warnings
        hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
        logger.debug(f"hidden_states shape: {hidden_states.shape}")
        labels = df_merged["definition"]  # Assume label column name is "label", please modify according to actual data

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Plot
        plot_hidden_states_layers(
            hidden_states, 
            labels=labels, 
            method='pca', 
            save_dir=args.output_dir, 
            show_plots=False
        )
        logger.info(f"Plots saved to directory: {args.output_dir}")

if __name__ == "__main__":
    main()