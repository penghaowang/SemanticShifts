import argparse
import torch
import os
import numpy as np
import logging

# Import functions written earlier from evaluate.py
from evaluate import merge_label, resolve_file_path
from plot import plot_hidden_states, fix_tensor_from_series
from logger_config import setup_logger

# Set up logging
logger = setup_logger('run_plot_layer_groups', 'logs/run_plot_layer_groups.log')

def plot_hidden_states_layer_group(
    hidden_states,
    labels=None,
    method='pca',
    save_dir='.',
    show_plots=False,
    df=None,
    content_column='generated_content',
    top_n=3,
    start_layer=0,
    end_layer=None,
    group_name=""
):
    """
    Reduce dimensions and plot scatter plots for hidden states of specified layers (optional labels).
    
    Args:
        hidden_states (torch.Tensor/np.ndarray): [samples, layers, beams, hidden_dim] or [samples, layers, hidden_dim]
        labels (list/ndarray): Label corresponding to each sample (length=samples). If None, no color differentiation.
        method (str): Dimensionality reduction method, default 'pca'.
        save_dir (str): Directory path to save plots.
        show_plots (bool): Whether to call plt.show() after the function ends.
        df (pandas.DataFrame, optional): DataFrame containing original text content, used to count the most common content for each label.
        content_column (str): Column name in df containing text content, default 'generated_content'.
        top_n (int): Number of most common contents to display for each label, default 3.
        start_layer (int): Start layer index (0-based)
        end_layer (int): End layer index (exclusive), if None, plot up to the last layer
        group_name (str): Name of the layer group, used for title and filename
    """
    # Convert to NumPy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    elif not isinstance(hidden_states, np.ndarray):
        raise ValueError("hidden_states must be a PyTorch Tensor or NumPy array.")

    if hidden_states.ndim not in [3,4]:
        raise ValueError("hidden_states dimensions must be [N, L, D] or [N, L, B, D].")

    # If 4D [samples, layers, beams, hidden_dim], select one based on beams first
    if hidden_states.ndim == 4:
        # For example, take only beam=0
        hidden_states = hidden_states[:, :, 0, :]

    # Now it's [samples, layers, hidden_dim]
    num_layers = hidden_states.shape[1]
    
    # Determine end layer index
    if end_layer is None or end_layer > num_layers:
        end_layer = num_layers
    
    # Create save directory
    group_dir = os.path.join(save_dir, f"layers_{start_layer+1}_to_{end_layer}")
    os.makedirs(group_dir, exist_ok=True)
    
    logger.info(f"Starting to plot charts for layers {start_layer+1} to {end_layer} ({group_name})")

    # Only plot layers within the specified range
    for layer_idx in range(start_layer, end_layer):
        layer_hs = hidden_states[:, layer_idx, :]  # [samples, hidden_dim]
        title = f'{group_name} - Layer {layer_idx+1} Scatter ({method.upper()})'
        save_path = os.path.join(group_dir, f'layer_{layer_idx+1}_{method}_scatter.png')

        plot_hidden_states(
            layer_hs,
            labels=labels,
            method=method,
            title=title,
            save_path=save_path,
            show_plot=show_plots,
            df=df,
            content_column=content_column,
            top_n=top_n
        )
    
    logger.info(f"Charts for layer group {group_name} saved to {group_dir}")


def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Plot scatter plots of hidden states for specified layer groups")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to the hidden states tensor file (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=False, help='Path to the sentence CSV file')
    parser.add_argument('--label_csv', type=str, required=False, help='Path to the label CSV file')
    parser.add_argument('--output_dir', type=str, default='layer_groups_output', help='Directory to save plots')
    parser.add_argument('--method', type=str, default='umap', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='Dimensionality reduction method')
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

    # Load hidden states data
    hidden_states = torch.load(hidden_states_file)
    logger.info(f"Loaded hidden states with shape: {hidden_states.shape}")

    # Define three layer groups
    layer_groups = [
        {"start": 0, "end": 10, "name": "Early Layers (1-10)"},
        {"start": 10, "end": 20, "name": "Middle Layers (11-20)"},
        {"start": 20, "end": 33, "name": "Later Layers (21-33)"}
    ]

    # Create root output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # If sentence_csv and label_csv are not provided, only plot hidden states (without label differentiation)
    if not args.sentence_csv or not args.label_csv:
        logger.info("sentence_csv and label_csv not provided. Will plot unlabeled scatter plot only.")
        
        # Generate plots for each layer group
        for group in layer_groups:
            plot_hidden_states_layer_group(
                hidden_states, 
                labels=None,  # No color differentiation
                method=args.method,
                save_dir=args.output_dir, 
                show_plots=False,
                start_layer=group["start"],
                end_layer=group["end"],
                group_name=group["name"]
            )
            
        logger.info(f"All layer group plots saved to directory: {args.output_dir}")
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
        labels = df_merged["definition"]  # Use 'definition' column as labels

        # Generate plots for each layer group
        for group in layer_groups:
            plot_hidden_states_layer_group(
                hidden_states, 
                labels=labels, 
                method=args.method, 
                save_dir=args.output_dir, 
                show_plots=False,
                df=df_merged,
                content_column='generated_content',
                top_n=3,
                start_layer=group["start"],
                end_layer=group["end"],
                group_name=group["name"]
            )
            
        logger.info(f"All layer group plots saved to directory: {args.output_dir}")

if __name__ == "__main__":
    main() 