import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, DBSCAN
from scipy.interpolate import make_interp_spline
import matplotlib.cm as cm
from evaluate import (
    reduce_dimensions, 
    find_most_rep, 
    prepare_hidden_states, 
    fix_tensor_from_series,
    compare_hidden_states_distributions,
    merge_label,
    get_time_periods,
    calculate_distribution_entropy,
    calculate_kl_divergence,
    cosine_similarity_layers
)

def plot_hidden_states(
    hidden_states, 
    labels=None,
    method='pca', 
    title='Hidden States Scatter Plot', 
    save_path=None, 
    show_plot=False,
    df=None,
    content_column='generated_content', 
    top_n=3
):
    """
    Reduce hidden states to 2D and plot as scatter.
    
    If labels provided, use different colors.
    If df provided, show most common content for each label.
    """
    # Convert to NumPy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    elif not isinstance(hidden_states, np.ndarray):
        raise ValueError("hidden_states must be PyTorch tensor or NumPy array.")

    # Dimension reduction
    reduced_vectors = reduce_dimensions(hidden_states, method=method, n_components=2)

    plt.figure(figsize=(7, 6))

    # Plot scatter
    if labels is None:
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7, c='blue')
    else:
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

    # Print most common content for each label
    if labels is not None and df is not None and content_column in df.columns:
        if len(df) == len(labels):
            from collections import Counter
            footer_texts = []
            unique_labels = sorted(set(labels))
            for lab in unique_labels:
                mask = (labels == lab)
                sub_df = df[mask]
                content_counts = Counter(sub_df[content_column])
                top_items = content_counts.most_common(top_n)
                if not top_items:
                    footer_texts.append(f"Label {lab}: (no content)")
                else:
                    items_str = " | ".join(
                        f"{txt}({cnt}x)" for txt, cnt in top_items
                    )
                    footer_texts.append(f"Label {lab}: {items_str}")

            text_block = "\n".join(footer_texts)
            plt.subplots_adjust(bottom=0.3)
            plt.figtext(0.01, 0.01, text_block, fontsize=8, va="bottom", ha="left")
        else:
            print("Warning: df rows and labels count mismatch, cannot show common content.")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_hidden_states_layers(
    hidden_states,
    labels=None,  
    method='pca',
    save_dir='.',
    show_plots=False,
    df=None,
    content_column='generated_content',
    top_n=3
):
    """
    Plot scatter for each layer's hidden states.
    
    Parameters:
        hidden_states: [samples, layers, beams, hidden_dim] or [samples, layers, hidden_dim]
        labels: Sample labels (length=samples), if None, no color distinction
        method: Dimension reduction method
        save_dir: Directory to save plots
        show_plots: Whether to call plt.show()
        df: DataFrame with original text content
        content_column: Column name with text content
        top_n: Number of most common contents to show per label
    """
    hs_np = prepare_hidden_states(hidden_states)

    num_layers = hs_np.shape[1]
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx in range(num_layers):
        layer_hs = hs_np[:, layer_idx, :]  # [samples, hidden_dim]
        title = f'Layer {layer_idx + 1} Scatter ({method.upper()})'
        save_path = os.path.join(save_dir, f'layer_{layer_idx + 1}_{method}_scatter.png')

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


def plot_hidden_states_clusters(
    hidden_states,
    n_clusters=5,
    method='pca',
    title='Unsupervised Clustering Scatter Plot',
    save_path=None,
    show_plot=False
):
    """
    Perform KMeans clustering on reduced hidden states and plot.

    Parameters:
        hidden_states: [N, hidden_dim]
        n_clusters: Number of clusters for KMeans
        method: Dimension reduction method
        title: Plot title
        save_path: Path to save plot
        show_plot: Whether to display plot
    """
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()

    reduced_vectors = reduce_dimensions(hidden_states, method=method, n_components=2)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_vectors)

    plt.figure(figsize=(6,5))
    cmap = plt.cm.get_cmap('rainbow', n_clusters)

    for c in range(n_clusters):
        idx = (cluster_labels == c)
        plt.scatter(
            reduced_vectors[idx, 0],
            reduced_vectors[idx, 1],
            color=cmap(c),
            alpha=0.7,
            label=f'Cluster {c}'
        )

    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_hidden_states_layers_clusters(
    hidden_states,
    n_clusters=5,
    method='pca',
    save_dir='.',
    show_plots=False
):
    """
    Plot KMeans clusters for each layer's hidden states.
    
    Parameters:
        hidden_states: [N, L, D] or [N, L, B, D]
        n_clusters: Number of KMeans clusters
        method: Dimension reduction method
        save_dir: Directory to save plots
        show_plots: Whether to call plt.show()
    """
    hs_np = prepare_hidden_states(hidden_states)

    num_layers = hs_np.shape[1]
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx in range(num_layers):
        layer_data = hs_np[:, layer_idx, :]
        layer_title = f'Layer {layer_idx + 1} Unsupervised Clustering ({method.upper()})'
        save_path = os.path.join(save_dir, f'layer_{layer_idx + 1}_{method}_clusters.png')
        
        plot_hidden_states_clusters(
            layer_data,
            n_clusters=n_clusters,
            method=method,
            title=layer_title,
            save_path=save_path,
            show_plot=show_plots
        )


def plot_hidden_states_spontaneous_clusters(
    hidden_states,
    df=None,
    content_column='generated_content',
    eps=0.5,
    min_samples=5,
    method='pca',
    title='Spontaneous Clustering (DBSCAN)',
    save_path=None,
    show_plot=False,
    top_n=3
):
    """
    Use DBSCAN for clustering (no predefined cluster count) and show most common content.

    Parameters:
        hidden_states: [N, hidden_dim]
        df: DataFrame with rows matching hidden_states
        content_column: Column with text content
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        method: Dimension reduction method
        title: Plot title
        save_path: Path to save plot
        show_plot: Whether to show plot
        top_n: Number of most common contents to show per cluster
    """
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()

    reduced_vectors = reduce_dimensions(hidden_states, method=method, n_components=2)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(reduced_vectors)

    plt.figure(figsize=(6,5))

    unique_clusters = np.unique(cluster_labels)
    cmap = plt.cm.get_cmap('rainbow', len(unique_clusters))

    cluster_info = []

    for i, c in enumerate(unique_clusters):
        idx = np.where(cluster_labels == c)[0]
        color = cmap(i)

        if c == -1:
            cluster_name = "Outliers"
        else:
            if df is not None and content_column in df.columns:
                sub_df = df.iloc[idx]
                contents = sub_df[content_column].tolist()
                rep_content = find_most_rep(contents)
                if not rep_content:
                    cluster_name = f"Cluster {c} (Empty?)"
                else:
                    cluster_name = f"Cluster {c}: {rep_content}"
                    
                    if c != -1:  # Ignore noise points
                        most_common = Counter(sub_df[content_column]).most_common(top_n)
                        cluster_info.append((c, most_common))
            else:
                cluster_name = f"Cluster {c}"

        plt.scatter(
            reduced_vectors[idx, 0],
            reduced_vectors[idx, 1],
            color=color,
            alpha=0.7,
            label=cluster_name
        )

    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    if cluster_info:
        footer_text = []
        for c, contents in cluster_info:
            text = [f"Cluster {c}:"]
            for cont, cnt in contents:
                text.append(f"  {cont} ({cnt})")
            footer_text.append("\n".join(text))
        
        plt.figtext(0.1, -0.1, "\n\n".join(footer_text), 
                   va="top", fontsize=8, wrap=True)
        plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_hidden_states_layers_spontaneous_clusters(
    hidden_states,
    df=None,
    content_column='generated_content',
    eps=0.5,
    min_samples=5,
    method='umap',
    save_dir='.',
    show_plots=False,
    top_n=3
):
    """
    Plot DBSCAN clusters for each layer's hidden states.
    
    Parameters:
        hidden_states: [N, L, (B), hidden_dim]
        df: DataFrame with matching samples
        content_column: Column with text content
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        method: Dimension reduction method
        save_dir: Directory to save plots
        show_plots: Whether to show plots
        top_n: Number of most common contents to show per cluster
    """
    hs_np = prepare_hidden_states(hidden_states)

    num_layers = hs_np.shape[1]
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx in range(num_layers):
        layer_data = hs_np[:, layer_idx, :]
        layer_title = f'Layer {layer_idx+1} - Spontaneous Clustering (DBSCAN / {method.upper()})'
        save_path = os.path.join(save_dir, f'layer_{layer_idx+1}_{method}_dbscan.png')

        plot_hidden_states_spontaneous_clusters(
            layer_data,
            df=df,
            content_column=content_column,
            eps=eps,
            min_samples=min_samples,
            method=method,
            title=layer_title,
            save_path=save_path,
            show_plot=show_plots,
            top_n=top_n
        )
def plot_all_layers_together_dbscan(
    hidden_states: torch.Tensor,
    df=None,
    content_column='generated_content',
    eps=0.5,
    min_samples=5,
    method='pca',
    save_path=None,
    show_plot=False,
    top_n=3
):
    """
    Flatten multiple layers into single feature vector and cluster with DBSCAN.
    """
    hs_np = prepare_hidden_states(hidden_states)
    
    N, L, D = hs_np.shape
    
    # Flatten: combine features from all layers => [N, L*D]
    flattened = hs_np.reshape(N, L * D)
    
    plot_hidden_states_spontaneous_clusters(
        flattened,
        df=df,
        content_column=content_column,
        eps=eps,
        min_samples=min_samples,
        method=method,
        title=f"DBSCAN on All Layers Flattened ({method.upper()})",
        save_path=save_path,
        show_plot=show_plot,
        top_n=top_n
    )
def plot_heatmap_layerwise(vec, save_path=None, show_plot=False):
    """
    Plot layer-wise heatmap
    vec shape: (n_samples, n_layers, n_layers)
    """
    #mean over samples
    vec = np.mean(vec, axis=0)
    #plot heatmap
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(vec, cmap='viridis')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()
def plot_relative_similarity(vec, save_path=None, show_plot=False):
    """
    Plot relative similarity between adjacent layers as bar chart
    vec shape: (n_samples, n_layers, n_layers)
    """
    #mean over samples
    vec = np.mean(vec, axis=0)
    n_layers = vec.shape[0]
    rel = np.zeros(n_layers - 1)
    for i in range(n_layers - 1):
        rel[i] = vec[i, i+1]
    #plot bar chart
    plt.figure(figsize=(10, 8))
    plt.bar(range(n_layers - 1), rel)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_word_trajectories(
    base_dir: str,
    output_dir: str,
    method: str = 'umap',
    layer_idx: int = -1,
    arrow_scale: float = 0.1,
    extraction_method: str = 'input_last_token',
    word_list: list = None,
    min_points: int = 3,
    max_labels_per_word: int = 2,
    logger = None
):
    """
    Load hidden states of multiple words and plot their trajectories in the same figure
    
    Parameters:
        base_dir (str): Base directory for hidden state files
        output_dir (str): Output directory
        method (str): Dimensionality reduction method, default 'umap'
        layer_idx (int): Layer index to use, -1 means use all layers
        arrow_scale (float): Arrow size scaling factor
        extraction_method (str): Extraction method, default 'input_last_token'
        word_list (list): List of words to include, None means all words
        min_points (int): Minimum number of points to plot trajectory, default 3
        max_labels_per_word (int): Maximum number of labels to display per word, default 2, 0 means no limit
        logger: Logger object, optional
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all word directories
    word_dirs = glob.glob(os.path.join(base_dir, "*"))
    # Filter out paths that are not word directories (typically word directories contain underscores)
    word_dirs = [d for d in word_dirs if os.path.basename(d).find('_') != -1]
    
    if logger:
        logger.info(f"Found {len(word_dirs)} word directories")
    
    if not word_dirs:
        if logger:
            logger.error(f"No matching word directories found in {base_dir}")
        return
    
    # If word list is provided, filter directories
    if word_list:
        filtered_dirs = []
        for word in word_list:
            # Find all directories containing the word (regardless of POS)
            for dir_path in word_dirs:
                dir_name = os.path.basename(dir_path)
                word_part = dir_name.split('_')[0]
                if word_part.lower() == word.lower():
                    filtered_dirs.append(dir_path)
        word_dirs = filtered_dirs
        if logger:
            logger.info(f"After filtering, {len(word_dirs)} word directories remain")
    
    # Create dictionary for storing all data
    all_hidden_states = []
    all_meta_data = []
    
    # Iterate through each word directory and load data
    for word_dir in word_dirs:
        dir_name = os.path.basename(word_dir)
        word_name = dir_name.split('_')[0]
        if logger:
            logger.info(f"Processing word: {word_name}")
        
        # Hidden states directory
        hs_dir_pattern = os.path.join(word_dir, extraction_method, "combined", "hidden_states_*")
        hs_dirs = glob.glob(hs_dir_pattern)
        if not hs_dirs:
            if logger:
                logger.warning(f"Hidden states directory not found: {hs_dir_pattern}")
            continue
        
        # Use the latest hidden states directory
        hs_dir = sorted(hs_dirs)[-1]
        
        # Find hidden states files
        pt_files = glob.glob(os.path.join(hs_dir, "*.pt"))
        if not pt_files:
            if logger:
                logger.warning(f"No .pt files found in {hs_dir}")
            continue
        
        # Sentence CSV file directory
        sentence_dir_pattern = os.path.join(word_dir, extraction_method, "combined", f"icl_basic_{extraction_method}_{os.path.basename(word_dir)}_*")
        sentence_dirs = glob.glob(sentence_dir_pattern)
        if not sentence_dirs:
            if logger:
                logger.warning(f"Sentence CSV directory not found: {sentence_dir_pattern}")
            continue
        
        # Use the latest sentence directory
        sentence_dir = sorted(sentence_dirs)[-1]
        sentence_csv = os.path.join(sentence_dir, f"icl_basic_{extraction_method}_{os.path.basename(word_dir)}.csv")
        
        # Label CSV file
        label_csv = os.path.join("/users/phwang/users/master_thesis2/labeled_data_o3_mini", f"{os.path.basename(word_dir)}_labeled.csv")
        
        if not os.path.exists(sentence_csv):
            if logger:
                logger.warning(f"Sentence CSV file does not exist: {sentence_csv}")
            continue
            
        if not os.path.exists(label_csv):
            if logger:
                logger.warning(f"Label CSV file does not exist: {label_csv}")
            continue
        
        # Merge data
        if logger:
            logger.info(f"Merging hidden states and label data for {word_name}")
        try:
            df_merged = merge_label(
                hs_pt_file=pt_files[0],
                sentence_csv_file=sentence_csv,
                label_csv_file=label_csv
            )
            
            # Filter invalid labels (retain to prevent label_index being -1)
            df_merged = df_merged[df_merged['label_index'] != -1]
            
            if df_merged.empty:
                if logger:
                    logger.warning(f"Merged result for {word_name} is empty, skipping")
                continue
                
            # Add word name column
            df_merged['word_name'] = word_name
            
            # Get time grouping data - directly use period mode
            time_periods = get_time_periods()
            df_merged['time_bin'] = None
            
            for period in time_periods:
                start_year, end_year = map(int, period.split('-'))
                mask = (df_merged['year'] >= start_year) & (df_merged['year'] <= end_year)
                df_merged.loc[mask, 'time_bin'] = period
            
            # Convert hidden states
            hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
            
            # If multi-layer hidden states, select specified layer or use all layers
            if hidden_states.dim() > 2:
                if layer_idx >= 0 and layer_idx < hidden_states.shape[1]:
                    hidden_states = hidden_states[:, layer_idx, :]
                    if logger:
                        logger.info(f"Using layer {layer_idx} hidden states")
                else:
                    # If format is [sample,layer,1,hidden_dim], remove dimension of 1
                    if hidden_states.dim() == 4 and hidden_states.shape[2] == 1:
                        hidden_states = hidden_states.squeeze(2)
                    
                    # Concatenate all layer hidden states instead of averaging
                    # Convert shape from [sample, layer, hidden_dim] to [sample, layer*hidden_dim]
                    batch_size, num_layers, hidden_dim = hidden_states.shape
                    hidden_states = hidden_states.reshape(batch_size, -1)
                    if logger:
                        logger.info(f"Using all layers of hidden states, shape from [{batch_size}, {num_layers}, {hidden_dim}] to [{batch_size}, {num_layers*hidden_dim}]")
            
            # Save hidden states and metadata
            all_hidden_states.append(hidden_states)
            
            # Remove hidden_states column to save memory
            df_merged.drop('hidden_states', axis=1, inplace=True)
            all_meta_data.append(df_merged)
            
            if logger:
                logger.info(f"Data processing for {word_name} completed, {len(df_merged)} rows")
            
        except Exception as e:
            if logger:
                logger.error(f"Error processing {word_name}: {str(e)}")
            continue
    
    # Merge all word data
    if not all_hidden_states:
        if logger:
            logger.error("No valid data found, cannot plot")
        return
        
    # Merge all hidden states
    combined_hidden_states = torch.cat(all_hidden_states, dim=0)
    combined_meta_data = pd.concat(all_meta_data, ignore_index=True)
    
    # Filter label_index of -1 again
    combined_meta_data = combined_meta_data[combined_meta_data['label_index'] != -1].reset_index(drop=True)
    
    if logger:
        logger.info(f"Merged result: {len(combined_meta_data)} rows of data, involving {combined_meta_data['word_name'].nunique()} words")
    
    # Reduce dimensions for all hidden states
    reduced_vectors = reduce_dimensions(combined_hidden_states, method=method, n_components=2)
    
    # Group by word and time to calculate average vectors, regardless of different definitions
    grouped_vectors = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(combined_meta_data.itertuples()):
        target_word = row.word_name
        time_bin = row.time_bin
        if pd.notna(time_bin) and pd.notna(target_word):
            grouped_vectors[target_word][time_bin].append(reduced_vectors[idx])
    
    # Calculate average vector for each group
    mean_vectors = defaultdict(dict)
    for target_word in grouped_vectors:
        for time_bin in grouped_vectors[target_word]:
            vectors = grouped_vectors[target_word][time_bin]
            if vectors:
                mean_vec = np.mean(vectors, axis=0)
                # Ensure mean_vec is at least a 2D array
                if not isinstance(mean_vec, np.ndarray) or mean_vec.ndim == 0:
                    mean_vec = np.array([mean_vec, 0.0])  # Add a default y-coordinate
                elif mean_vec.size == 1:
                    mean_vec = np.array([mean_vec.item(), 0.0])  # Convert to 2D coordinate
                mean_vectors[target_word][time_bin] = mean_vec
    
    # Plot trajectory
    plt.figure(figsize=(16, 14))
    
    # Choose different base colors for each word
    words = list(mean_vectors.keys())
    word_colors = cm.tab20(np.linspace(0, 1, len(words)))
    
    # Collect labels and corresponding lines for legend
    legend_elements = []
    
    # Iterate through each word
    for word_idx, target_word in enumerate(words):
        # Sort by time
        time_bins = sorted(mean_vectors[target_word].keys())
        
        if len(time_bins) < min_points:
            if logger:
                logger.warning(f"Word '{target_word}' has only {len(time_bins)} time points, need at least {min_points} to plot trajectory")
            continue
        
        # Extract coordinates
        x_coords = []
        y_coords = []
        for t in time_bins:
            vec = mean_vectors[target_word][t]
            # Ensure vector has at least two elements for x and y coordinates
            if isinstance(vec, np.ndarray) and vec.size >= 2:
                x_coords.append(vec[0])
                y_coords.append(vec[1])
            else:
                if logger:
                    logger.warning(f"Word '{target_word}' at time '{t}' has insufficient vector dimensions, skipping this point")
                continue
        
        # If not enough valid coordinate points, skip plotting
        if len(x_coords) < min_points:
            if logger:
                logger.warning(f"Word '{target_word}' does not have enough valid coordinate points, cannot plot trajectory")
            continue
        
        # Use word as legend label
        legend_label = f"{target_word}"
        
        # Use spline interpolation to generate smooth curve
        if len(x_coords) >= 4:  # At least 4 points needed for cubic spline interpolation
            # Generate denser points for smooth curve
            t_points = np.linspace(0, 1, len(x_coords))
            t_smooth = np.linspace(0, 1, 100)  # 100 points for smooth curve
            
            # Create spline objects
            spl_x = make_interp_spline(t_points, x_coords, k=3)
            spl_y = make_interp_spline(t_points, y_coords, k=3)
            
            # Calculate points for smooth curve
            x_smooth = spl_x(t_smooth)
            y_smooth = spl_y(t_smooth)
            
            # Plot smooth curve
            line, = plt.plot(x_smooth, y_smooth, '-', color=word_colors[word_idx], 
                         label=legend_label, alpha=0.7, linewidth=2)
            # Plot small dots on original data points
            plt.plot(x_coords, y_coords, 'o', color=word_colors[word_idx], markersize=4, alpha=0.7)
            
            # Use smooth curve to draw arrows
            arrow_indices = np.linspace(10, 90, 5, dtype=int)  # Choose 5 evenly distributed points on smooth curve
            for i in range(len(arrow_indices) - 1):
                idx1 = arrow_indices[i]
                idx2 = arrow_indices[i + 1]
                dx = x_smooth[idx2] - x_smooth[idx1]
                dy = y_smooth[idx2] - y_smooth[idx1]
                # Arrow position between two points
                arrow_x = x_smooth[idx1 + (idx2-idx1)//2]  # Use point directly on curve
                arrow_y = y_smooth[idx1 + (idx2-idx1)//2]
                
                # Calculate direction vector and normalize
                length = np.sqrt(dx**2 + dy**2)
                norm_dx = dx / length if length > 0 else 0
                norm_dy = dy / length if length > 0 else 0
                
                # Use FancyArrow to draw arrow directly on curve
                plt.arrow(arrow_x, arrow_y, norm_dx * arrow_scale * 2, norm_dy * arrow_scale * 2,
                          head_width=arrow_scale, head_length=arrow_scale,
                          fc=word_colors[word_idx], ec=word_colors[word_idx], alpha=0.8,
                          overhang=0.3, length_includes_head=True)
        else:
            # If too few points, use original line
            line, = plt.plot(x_coords, y_coords, '-o', color=word_colors[word_idx], 
                         label=legend_label, alpha=0.7, linewidth=2)
            
            # Add arrows to indicate time direction
            for j in range(len(time_bins) - 1):
                dx = x_coords[j+1] - x_coords[j]
                dy = y_coords[j+1] - y_coords[j]
                # Add arrows at points on the segment, not outside the midpoint
                mid_x = (x_coords[j] + x_coords[j+1]) / 2
                mid_y = (y_coords[j] + y_coords[j+1]) / 2
                
                # Calculate direction vector and normalize
                length = np.sqrt(dx**2 + dy**2)
                norm_dx = dx / length if length > 0 else 0
                norm_dy = dy / length if length > 0 else 0
                
                # Use FancyArrow to draw arrow directly on curve
                plt.arrow(mid_x, mid_y, norm_dx * arrow_scale * 2, norm_dy * arrow_scale * 2,
                          head_width=arrow_scale, head_length=arrow_scale,
                          fc=word_colors[word_idx], ec=word_colors[word_idx], alpha=0.8,
                          overhang=0.3, length_includes_head=True)
        
        legend_elements.append(line)
        
        # Draw special markers for start and end points
        plt.scatter(x_coords[0], y_coords[0], color=word_colors[word_idx], s=100, marker='*')  # Start point
        plt.scatter(x_coords[-1], y_coords[-1], color=word_colors[word_idx], s=100, marker='s')  # End point
        
        # Add word annotation at center of curve, not year
        mid_idx = len(x_coords) // 2
        plt.annotate(f"{target_word}", 
                   (x_coords[mid_idx], y_coords[mid_idx]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=12, color=word_colors[word_idx], 
                   fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                                              fc="white", ec=word_colors[word_idx], alpha=0.7))
    
    # Set up chart
    title = f"Semantic Trajectory of Words ({method.upper()})"
    
    plt.title(title, fontsize=18)
    plt.xlabel("Component 1", fontsize=14)
    plt.ylabel("Component 2", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, title="Words", bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize='medium', ncol=2, title_fontsize=14)
    
    # Increase tick label font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save chart
    save_path = os.path.join(output_dir, f"all_words_combined_{method}_{extraction_method}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if logger:
        logger.info(f"Chart saved to {save_path}")
    plt.close()

def plot_correlation_heatmap(correlation_matrix: np.ndarray, method_names: list, 
                            title: str, output_file: str):
    """
    Plot correlation matrix as heatmap.
    
    Parameters:
        correlation_matrix: Square matrix of correlations
        method_names: Names of methods to use as labels
        title: Plot title
        output_file: Path to save output figure
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        correlation_matrix, 
        annot=True,  # Show values
        fmt=".3f",  # Value format
        cmap="YlGnBu",  # Color map
        xticklabels=method_names,
        yticklabels=method_names,
        vmin=0.0,  # Minimum value
        vmax=1.0,  # Maximum value
        cbar_kws={'label': 'Cosine Similarity'}  # Colorbar label
    )
    
    # Set title and save plot
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    plt.close()

def plot_semantic_change(
    hidden_states_dir: str,
    sentence_csv: str,
    label_csv: str,
    output_dir: str,
    method: str = 'umap',
    layer_idx: int = -1,
    word: str = None,
    pos: str = None,
    time_bin_level: str = 'period',  # 'year' or 'period'
    logger = None
):
    """
    Plot semantic changes of words over time, including KL divergence and entropy changes
    
    Parameters:
        hidden_states_dir: Directory containing hidden state files
        sentence_csv: Path to sentence CSV file
        label_csv: Path to label CSV file
        output_dir: Output directory
        method: Dimensionality reduction method, default 'umap'
        layer_idx: Layer index to use, -1 means average of all layers
        word: Target word, if None all words are used
        pos: Part of speech tag, if None no POS filtering is applied
        time_bin_level: Time grouping level, 'year' or 'period'
        logger: Logger object, optional
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find hidden states file
    hidden_states_file = None
    for file in os.listdir(hidden_states_dir):
        if file.endswith('.pt'):
            hidden_states_file = os.path.join(hidden_states_dir, file)
            break
    
    if not hidden_states_file:
        if logger:
            logger.error(f"No hidden states file found in {hidden_states_dir}")
        return
        
    # Merge data
    if logger:
        logger.info("Starting to merge hidden states and label data")
    
    df_merged = merge_label(
        hs_pt_file=hidden_states_file,
        sentence_csv_file=sentence_csv,
        label_csv_file=label_csv
    )
    
    # Filter invalid labels
    df_merged = df_merged[df_merged['label_index'] != -1]
    if df_merged.empty:
        if logger:
            logger.error("Merged result is empty, cannot plot")
        return
    
    if logger:
        logger.info(f"Merge completed, total {len(df_merged)} rows of data")
    
    # Filter by specified word and POS (if provided)
    if word:
        df_merged = df_merged[df_merged['target_word'].str.lower() == word.lower()]
        if df_merged.empty:
            if logger:
                logger.error(f"No matching word found: {word}")
            return
    
    if pos:
        df_merged = df_merged[df_merged['target_pos'] == pos]
        if df_merged.empty:
            if logger:
                logger.error(f"No matching POS found: {pos}")
            return
    
    # Get time grouping data
    time_periods = None
    if time_bin_level == 'period':
        time_periods = get_time_periods()
        df_merged['time_bin'] = None
        
        for period in time_periods:
            start_year, end_year = map(int, period.split('-'))
            mask = (df_merged['year'] >= start_year) & (df_merged['year'] <= end_year)
            df_merged.loc[mask, 'time_bin'] = period
    else:  # Use year as time bin
        df_merged['time_bin'] = df_merged['year']
    
    # Convert hidden states
    hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
    
    # Process hidden states
    if hidden_states.dim() == 4:
        num_layers = hidden_states.shape[1]
        if num_layers >= 12:
            start_layer = num_layers - 12
            hidden_states = hidden_states[:, start_layer:, 0, :]
        else:
            hidden_states = hidden_states[:, :, 0, :]
        hidden_states = hidden_states.mean(dim=1)
    
    elif hidden_states.dim() == 3:
        num_layers = hidden_states.shape[1]
        if num_layers >= 12:
            start_layer = num_layers - 12
            hidden_states = hidden_states[:, start_layer:, :]
        hidden_states = hidden_states.mean(dim=1)
    
    # Reduce to 16 dimensions for all vectors
    if hidden_states.shape[1] > 16:
        hidden_states_np = hidden_states.cpu().numpy()
        reduced_hidden_states = reduce_dimensions(hidden_states_np, method='pca', n_components=16)
        hidden_states = torch.tensor(reduced_hidden_states)
    
    # Reset index to ensure correct masking
    df_merged = df_merged.reset_index(drop=True)
    
    # Check if polysemous (has multiple labels)
    unique_labels = df_merged['label_index'].unique()
    is_polysemous = len(unique_labels) > 1
    
    # Collect hidden states by time bin
    time_bins = sorted(df_merged['time_bin'].unique())
    time_bins_vectors = {}
    time_bins_labels = {}
    
    # If polysemous, also collect vectors by label
    if is_polysemous:
        time_bins_label_vectors = {}
    
    for time_bin in time_bins:
        if pd.isna(time_bin):
            continue
            
        mask = df_merged['time_bin'] == time_bin
        indices = df_merged[mask].index.tolist()
        
        if indices:
            time_vectors = hidden_states[indices].numpy()
            if time_vectors.ndim > 2:
                time_vectors = np.mean(time_vectors, axis=1)
            
            time_labels = df_merged.loc[indices, 'label_index'].values
            
            time_bins_vectors[time_bin] = time_vectors
            time_bins_labels[time_bin] = time_labels
            
            # If polysemous, collect vectors by label
            if is_polysemous:
                time_bins_label_vectors[time_bin] = {}
                time_df = df_merged.loc[indices]
                
                for label in unique_labels:
                    label_mask = time_df['label_index'] == label
                    if sum(label_mask) > 0:
                        label_indices = time_df[label_mask].index.tolist()
                        label_vectors = hidden_states[label_indices].numpy()
                        
                        if label_vectors.ndim > 2:
                            label_vectors = np.mean(label_vectors, axis=1)
                        
                        if len(label_vectors) >= 3:
                            time_bins_label_vectors[time_bin][label] = label_vectors
    
    # Calculate KL divergence between adjacent time bins
    kl_divergences = []
    for i in range(len(time_bins) - 1):
        current_bin = time_bins[i]
        next_bin = time_bins[i+1]
        
        if current_bin in time_bins_vectors and next_bin in time_bins_vectors:
            vectors1 = time_bins_vectors[current_bin]
            vectors2 = time_bins_vectors[next_bin]
            

            distribution_metrics = compare_hidden_states_distributions(vectors1, vectors2)
            kl_1_to_2 = distribution_metrics.get('avg_kl_1_to_2', float('nan'))
            kl_2_to_1 = distribution_metrics.get('avg_kl_2_to_1', float('nan'))
            js_div = distribution_metrics.get('avg_js', float('nan'))
            
            kl_divergences.append({
                'time_pair_index': i,
                'time_pair': f"{current_bin} → {next_bin}",
                'current_bin': current_bin,
                'next_bin': next_bin,
                'kl_1_to_2': kl_1_to_2,
                'kl_2_to_1': kl_2_to_1,
                'js_div': js_div
            })

    # Plot KL divergence over time
    plt.figure(figsize=(10, 6))
    x = range(len(kl_divergences))
    
    kl_1_to_2_values = [d['kl_1_to_2'] for d in kl_divergences]
    kl_2_to_1_values = [d['kl_2_to_1'] for d in kl_divergences]
    js_values = [d['js_div'] for d in kl_divergences]
    
    plt.plot(x, kl_1_to_2_values, 'o-', label='KL(t → t+1)', color='blue')
    plt.plot(x, kl_2_to_1_values, 's-', label='KL(t+1 → t)', color='red')
    plt.plot(x, js_values, '^-', label='Jensen-Shannon', color='green')
    
    plt.xticks(x, [d['time_pair'] for d in kl_divergences], rotation=45)
    plt.xlabel('Time Period Transitions')
    plt.ylabel('Divergence Value')
    plt.title('Semantic Divergence Between Time Periods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    kl_plot_path = os.path.join(output_dir, 'kl_divergence_over_time.png')
    plt.savefig(kl_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate scatter plots of word usage across time periods
    for time_bin in time_bins:
        if time_bin not in time_bins_vectors:
            continue
            
        vectors = time_bins_vectors[time_bin]
        labels = time_bins_labels[time_bin]
        
        # Reduce dimensionality for visualization
        reduced_vectors = reduce_dimensions(vectors, method=method, n_components=2)
        
        plt.figure(figsize=(9, 7))
        
        if is_polysemous:
            # Plot each sense with different color
            unique_sense_labels = sorted(set(labels))
            cmap = plt.cm.get_cmap('rainbow', len(unique_sense_labels))
            
            for i, label in enumerate(unique_sense_labels):
                mask = labels == label
                plt.scatter(
                    reduced_vectors[mask, 0], 
                    reduced_vectors[mask, 1], 
                    color=cmap(i),
                    label=f"Sense {label}",
                    alpha=0.7
                )
                
            plt.legend(title="Word Senses")
        else:
            # If only one sense, use single color
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)
        
        # Add title and labels
        word_info = f"'{word}'" if word else "All Words"
        pos_info = f"(POS: {pos})" if pos else ""
        plt.title(f"Semantic Space for {word_info} {pos_info}\nPeriod: {time_bin}")
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        period_plot_path = os.path.join(output_dir, f'semantic_space_{time_bin}.png')
        plt.savefig(period_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    if logger:
        logger.info("Semantic change plotting completed")
