import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.cm as cm
import glob
from scipy.interpolate import make_interp_spline

from evaluate import reduce_dimensions, merge_label, get_time_periods
from plot import fix_tensor_from_series, plot_word_trajectories
from logger_config import setup_logger

# Set up logging
logger = setup_logger('plot_word_trajectory', 'logs/plot_word_trajectory.log')

def main():
    """Process command line arguments and call plotting functions"""
    parser = argparse.ArgumentParser(description='Plot word trajectories over time')
    
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing word directories')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for plots')
    parser.add_argument('--method', type=str, default='umap',
                        choices=['pca', 'tsne', 'umap'],
                        help='Dimensionality reduction method')
    parser.add_argument('--layer_idx', type=int, default=-1,
                        help='Layer index to use, -1 means use all layers')
    parser.add_argument('--arrow_scale', type=float, default=0.1,
                        help='Scale factor for arrows in trajectory plot')
    parser.add_argument('--extraction_method', type=str, 
                        default='input_last_token',
                        help='Extraction method used for hidden states')
    parser.add_argument('--words', type=str, default=None,
                        help='Comma-separated list of words to include; if not provided, all words are used')
    parser.add_argument('--min_points', type=int, default=3,
                        help='Minimum number of time points needed to plot a word trajectory')
    parser.add_argument('--max_labels', type=int, default=2,
                        help='Maximum number of labels to show per word (0 for no limit)')
    
    args = parser.parse_args()
    
    # Convert comma-separated words to list
    word_list = None
    if args.words:
        word_list = [w.strip() for w in args.words.split(',')]
    
    # Call the function from plot.py
    plot_word_trajectories(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        method=args.method,
        layer_idx=args.layer_idx,
        arrow_scale=args.arrow_scale,
        extraction_method=args.extraction_method,
        word_list=word_list,
        min_points=args.min_points,
        max_labels_per_word=args.max_labels,
        logger=logger
    )

if __name__ == "__main__":
    main() 