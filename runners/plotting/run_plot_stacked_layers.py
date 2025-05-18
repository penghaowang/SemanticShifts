import argparse
import torch
import os
import numpy as np
import logging
import matplotlib.pyplot as plt

# Import existing functions from evaluate.py
from evaluate import merge_label, resolve_file_path, reduce_dimensions
from plot import fix_tensor_from_series
from logger_config import setup_logger

# Set up logging
logger = setup_logger('run_plot_stacked_layers', 'logs/run_plot_stacked_layers.log')

def plot_stacked_layers(
    hidden_states,
    labels=None,
    method='umap',
    title='Stacked Layers',
    save_path=None,
    show_plot=False,
    df=None,
    content_column='generated_content',
    top_n=3
):
    """
    Stack hidden states from multiple layers, reduce dimensions, and plot a scatter plot.
    
    Args:
        hidden_states (torch.Tensor/np.ndarray): [samples, features] Stacked hidden states
        labels (list/ndarray): Label corresponding to each sample (length=samples). If None, no color differentiation.
        method (str): Dimensionality reduction method, default 'umap'.
        title (str): Plot title
        save_path (str): Path to save the plot
        show_plot (bool): Whether to display the plot
        df (pandas.DataFrame, optional): DataFrame containing original text content, used to count the most common content for each label.
        content_column (str): Column name in df containing text content, default 'generated_content'.
        top_n (int): Number of most common contents to display for each label, default 3.
    """
    # Convert to NumPy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    elif not isinstance(hidden_states, np.ndarray):
        raise ValueError("hidden_states must be a PyTorch Tensor or NumPy array.")

    # Reduce dimensions to 2D
    reduced_vectors = reduce_dimensions(hidden_states, method=method, n_components=2)

    plt.figure(figsize=(8, 6))

    # If labels are not provided, use a single color
    if labels is None:
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7, c='blue')
    else:
        # With labels, plot scatter points for each unique label separately with a legend
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap('rainbow', len(unique_labels))

        for i, lab in enumerate(unique_labels):
            idx = [j for j, x in enumerate(labels) if x == lab]
            plt.scatter(
                reduced_vectors[idx, 0],
                reduced_vectors[idx, 1],
                color=cmap(i),
                label=str(lab),
                alpha=0.7
            )
        plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    # Print the most common content for each label (if df is provided)
    if labels is not None and df is not None and content_column in df.columns:
        if len(df) == len(labels):
            from collections import Counter
            footer_texts = []
            unique_labels = sorted(set(labels))
            for lab in unique_labels:
                # Filter rows corresponding to the label
                mask = (labels == lab)
                sub_df = df[mask]
                # Count the top_n most common
                content_counts = Counter(sub_df[content_column])
                top_items = content_counts.most_common(top_n)
                if not top_items:
                    footer_texts.append(f"Label {lab}: (No content)")
                else:
                    # Create a short string
                    items_str = " | ".join(
                        f"{txt}({cnt}x)" for txt, cnt in top_items
                    )
                    footer_texts.append(f"Label {lab}: {items_str}")

            # Place these statistics below the plot
            text_block = "\n".join(footer_texts)
            plt.subplots_adjust(bottom=0.3)
            plt.figtext(0.01, 0.01, text_block, fontsize=8, va="bottom", ha="left")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def stack_layer_group(hidden_states, start_layer, end_layer):
    """
    Stack hidden states from a specified range of layers together
    
    Args:
        hidden_states (torch.Tensor/np.ndarray): [samples, layers, hidden_dim] or [samples, layers, beams, hidden_dim]
        start_layer (int): Start layer index (0-based)
        end_layer (int): End layer index (exclusive)
        
    Returns:
        Stacked hidden states [samples, (end_layer-start_layer)*hidden_dim]
    """
    # Convert to NumPy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    
    # If 4D [samples, layers, beams, hidden_dim], take beam=0 first
    if hidden_states.ndim == 4:
        hidden_states = hidden_states[:, :, 0, :]
    
    # Extract the specified range of layers
    layers_subset = hidden_states[:, start_layer:end_layer, :]  # [samples, selected_layers, hidden_dim]
    
    # Get shape information
    samples, selected_layers, hidden_dim = layers_subset.shape
    
    # Reshape tensor and stack layers
    # From [samples, selected_layers, hidden_dim] to [samples, selected_layers*hidden_dim]
    stacked_features = layers_subset.reshape(samples, selected_layers * hidden_dim)
    
    return stacked_features

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Stack hidden states from layer groups and plot scatter plots")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='Path to the hidden states tensor file (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=False, help='Path to the sentence CSV file')
    parser.add_argument('--label_csv', type=str, required=False, help='Path to the label CSV file')
    parser.add_argument('--output_dir', type=str, default='stacked_layers_output', help='Directory to save plots')
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # If sentence_csv and label_csv are not provided, only plot hidden states (without label differentiation)
    if not args.sentence_csv or not args.label_csv:
        logger.info("sentence_csv and label_csv not provided. Will plot unlabeled scatter plot only.")
        
        # Generate stacked plots for each layer group
        for group in layer_groups:
            # Stack all layers in the group
            stacked_features = stack_layer_group(
                hidden_states, 
                start_layer=group["start"], 
                end_layer=group["end"]
            )
            
            # Generate plot
            save_path = os.path.join(args.output_dir, f"stacked_{group['name'].replace(' ', '_').lower()}.png")
            plot_stacked_layers(
                stacked_features,
                labels=None,  # No color differentiation
                method=args.method,
                title=f"{group['name']} - Stacked ({args.method.upper()})",
                save_path=save_path,
                show_plot=False
            )
            
        logger.info(f"All layer group stacked plots saved to directory: {args.output_dir}")
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

        # Generate stacked plots for each layer group
        for group in layer_groups:
            # Stack all layers in the group
            stacked_features = stack_layer_group(
                hidden_states, 
                start_layer=group["start"], 
                end_layer=group["end"]
            )
            
            # Generate plot
            save_path = os.path.join(args.output_dir, f"stacked_{group['name'].replace(' ', '_').lower()}.png")
            plot_stacked_layers(
                stacked_features,
                labels=labels,
                method=args.method,
                title=f"{group['name']} - Stacked ({args.method.upper()})",
                save_path=save_path,
                show_plot=False,
                df=df_merged,
                content_column='generated_content',
                top_n=3
            )
            
        logger.info(f"All layer group stacked plots saved to directory: {args.output_dir}")

if __name__ == "__main__":
    main() 