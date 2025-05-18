import argparse
import torch
import os
import numpy as np
import logging
import pandas as pd
import traceback

# Import functions written earlier from evaluate.py
from evaluate import merge_label, resolve_file_path, get_time_periods
from plot import plot_hidden_states_layers, fix_tensor_from_series, plot_hidden_states
from logger_config import setup_logger

# Set up logging
logger = setup_logger('run_plot_comb_layers', 'logs/run_plot_comb_layers.log')

def parse_layers_str(layers_str):
    """
    Parse layer string, supporting comma separation and range notation
    Example: "0,1,2" or "0-5" or "0,1,5-7"
    
    Args:
        layers_str (str): Layer string
        
    Returns:
        list: List of layer indices
    """
    if not layers_str:
        return []
    
    layers = []
    parts = layers_str.split(',')
    
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    
    return sorted(list(set(layers)))  # Remove duplicates and sort

def main():
    try:
        # Define command line arguments
        parser = argparse.ArgumentParser(description="Run plot_hidden_states_layers with hidden states and optional labels")
        parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to hidden states tensor file (.pt)')
        parser.add_argument('--sentence_csv', type=str, required=True, help='Path to the sentence CSV file')
        parser.add_argument('--label_csv', type=str, required=True, help='Path to the label CSV file')
        parser.add_argument('--output_dir', type=str, default='plot_output', help='Where to save the plots')
        parser.add_argument('--method', type=str, default='umap', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='Dimensionality reduction method')
        parser.add_argument('--layers', type=str, required=True, help='Layers to analyze, e.g., "0,1,2" or "0-5"')
        parser.add_argument('--word', type=str, required=False, help='Target word')
        parser.add_argument('--extraction_method', type=str, required=False, help='Extraction method')
        args = parser.parse_args()

        logger.info(f"Starting processing: Word={args.word}, Extraction Method={args.extraction_method}, Layers={args.layers}, Reduction Method={args.method}")
        
        # Find hidden states file
        hidden_states_file = resolve_file_path(
            args.hidden_states_dir,
            "*.pt", 
            f"Hidden states file not found: {args.hidden_states_dir}",
            logger
        )
        
        if hidden_states_file is None:
            logger.error(f"Hidden states file not found: {args.hidden_states_dir}")
            return 1

        # Check if necessary CSV files exist
        if not os.path.exists(args.sentence_csv):
            logger.error(f"Sentence CSV file does not exist: {args.sentence_csv}")
            return 1
        
        if not os.path.exists(args.label_csv):
            logger.error(f"Label CSV file does not exist: {args.label_csv}")
            return 1

        # Execute merge_label
        logger.info("Starting to merge label data.")
        df_merged = merge_label(
            hs_pt_file=hidden_states_file,
            sentence_csv_file=args.sentence_csv,
            label_csv_file=args.label_csv
        )
        
        logger.info(f"Data size before merge: {len(df_merged)}, Rows with label -1: {sum(df_merged['label_index'] == -1)}")
        
        # filter out -1 and -3 labels
        df_merged = df_merged[df_merged['label_index'] != -1]
        df_merged = df_merged[df_merged['label_index'] != -3]

        
        logger.info(f"Data size after filtering label=-1 and label=-3: {len(df_merged)}")
        
        if df_merged.empty:
            logger.error("Merge result is empty, cannot plot. Please check label data.")
            return 1

        logger.info(f"Merge completed, DataFrame rows: {len(df_merged)}")
        
        # Analyze label distribution
        label_counts = df_merged['definition'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        # Parse layers to analyze
        layers = parse_layers_str(args.layers)
        if not layers:
            logger.error(f"Failed to parse layer argument: {args.layers}")
            return 1
        
        logger.info(f"Will analyze the following layers: {layers}")
        
        # Get hidden states
        try:
            hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
            logger.info(f"Hidden states tensor shape: {hidden_states.shape}")
        except Exception as e:
            logger.error(f"Failed to process hidden states: {str(e)}")
            return 1
        
        labels = df_merged["definition"]
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Extract hidden states for specified layers
        num_layers = hidden_states.shape[1]
        selected_layers = [l for l in layers if l < num_layers]
        
        if not selected_layers:
            logger.error(f"Specified layers {layers} are out of range for hidden states (0-{num_layers-1}).")
            return 1
        
        logger.info(f"Valid layers: {selected_layers}")
        
        # Combine hidden states from all selected layers
        combined_hidden_states = hidden_states[:, selected_layers, :]
        # Convert 3D tensor to 2D: concatenate features from all selected layers
        combined_hidden_states = combined_hidden_states.reshape(combined_hidden_states.shape[0], -1)
        
        logger.info(f"Shape of combined hidden states tensor: {combined_hidden_states.shape}")
        
        save_path = os.path.join(args.output_dir, f'combined_layers_{args.method}.png')
        logger.info(f"Plotting combined layers chart: {selected_layers}")
        
        plot_hidden_states(
            combined_hidden_states,
            labels=df_merged['definition'],
            method=args.method,
            title=f'Hidden States ({args.method}) - Layers {args.layers}',
            save_path=save_path,
            show_plot=False,
            df=df_merged,
            content_column='definition',
            top_n=3
        )
        
        # Verify if the file was created successfully
        if os.path.exists(save_path):
            logger.info(f"Chart saved successfully to: {save_path}")
        else:
            logger.error(f"Chart might not have been saved successfully, file does not exist: {save_path}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)