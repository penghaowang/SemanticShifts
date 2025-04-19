import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import matplotlib as mpl

from evaluate import (
    calculate_silhouette_score, 
    calculate_kl_divergence, 
    compare_hidden_states_distributions, 
    reduce_dimensions,
    merge_label,
    get_time_periods,
    calculate_distribution_entropy
)
from plot import fix_tensor_from_series, plot_semantic_change
from logger_config import setup_logger

# Set up logger
logger = setup_logger('plot_semantic_change', 'logs/plot_semantic_change.log')

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Plot semantic change over time")
    parser.add_argument("--hidden_states_dir", required=True, help="Directory containing hidden state files")
    parser.add_argument("--sentence_csv", required=True, help="Path to sentence CSV file")
    parser.add_argument("--label_csv", required=True, help="Path to label CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--method", default='umap', choices=['pca', 'tsne', 'umap'], help="Dimensionality reduction method")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index to use, -1 means average of all layers")
    parser.add_argument("--word", default=None, help="Target word, if None all words are used")
    parser.add_argument("--pos", default=None, help="Part of speech tag")
    parser.add_argument("--time_bin_level", default='period', choices=['year', 'period'], help="Time grouping level")
    
    args = parser.parse_args()
    
    # Call the plotting function from plot.py
    plot_semantic_change(
        hidden_states_dir=args.hidden_states_dir,
        sentence_csv=args.sentence_csv,
        label_csv=args.label_csv,
        output_dir=args.output_dir,
        method=args.method,
        layer_idx=args.layer_idx,
        word=args.word,
        pos=args.pos,
        time_bin_level=args.time_bin_level,
        logger=logger
    )

if __name__ == "__main__":
    main()