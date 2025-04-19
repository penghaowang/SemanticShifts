import os
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from tqdm import tqdm
from typing import Optional
import traceback
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
from collections import Counter
import glob

# Import custom logger
from logger_config import setup_logger, TRACE

# Set module logger
logger = setup_logger('evaluate', 'logs/evaluate.log')

def fix_tensor_from_series(series):
    """
    Convert Pandas Series with NumPy arrays to PyTorch Tensor efficiently.
    Avoids "Creating a tensor from a list of numpy.ndarrays is extremely slow" warning.
    """
    arr_list = series.values
    stacked = np.stack(arr_list, axis=0)
    return torch.from_numpy(stacked)

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> Tuple[float, float, float]:
    """
    Calculate KL divergence and Jensen-Shannon divergence between two distributions.
    
    Note: KL divergence is asymmetric (KL(p||q) != KL(q||p))
    Jensen-Shannon divergence is symmetric.
    
    Returns: (KL(p||q), KL(q||p), JS(p,q))
    """
    try:
        # Ensure inputs are numpy arrays
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        
        # Ensure p and q are probability distributions (sum to 1)
        if abs(np.sum(p) - 1.0) > epsilon:
            p = p / np.sum(p)
        if abs(np.sum(q) - 1.0) > epsilon:
            q = q / np.sum(q)
        
        # Add epsilon to prevent log(0)
        p = p + epsilon
        q = q + epsilon
        
        # Renormalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL(p||q)
        kl_pq = entropy(p, q)
        
        # Calculate KL(q||p)
        kl_qp = entropy(q, p)
        
        # Calculate Jensen-Shannon divergence
        js = jensenshannon(p, q, base=2)  # Base 2 makes JS range [0,1]
        js = js ** 2  # jensenshannon returns square root of divergence
        
        logger.debug(f"KL(p||q): {kl_pq:.4f}, KL(q||p): {kl_qp:.4f}, JS(p,q): {js:.4f}")
        return (kl_pq, kl_qp, js)
    except Exception as e:
        logger.error(f"KL divergence calculation error: {e}")
        logger.debug(traceback.format_exc())
        return (float('nan'), float('nan'), float('nan'))

def compare_hidden_states_distributions(hidden_states1: np.ndarray, 
                                       hidden_states2: np.ndarray, 
                                       n_bins: int = 50) -> Dict[str, float]:
    """
    Compare distribution differences between two sets of hidden states.
    """
    try:
        # Ensure inputs are numpy arrays
        if isinstance(hidden_states1, torch.Tensor):
            hidden_states1 = hidden_states1.cpu().numpy()
        if isinstance(hidden_states2, torch.Tensor):
            hidden_states2 = hidden_states2.cpu().numpy()
            
        # Calculate distribution differences for each dimension, then average
        hidden_dim = hidden_states1.shape[1]
        kl_pq_list, kl_qp_list, js_list = [], [], []
        
        # Calculate distribution distances for each dimension
        for dim in range(hidden_dim):
            # Extract values for current dimension
            values1 = hidden_states1[:, dim]
            values2 = hidden_states2[:, dim]
            
            # Calculate histograms to get distributions
            min_val = min(np.min(values1), np.min(values2))
            max_val = max(np.max(values1), np.max(values2))
            bins = np.linspace(min_val, max_val, n_bins+1)
            
            hist1, _ = np.histogram(values1, bins=bins, density=True)
            hist2, _ = np.histogram(values2, bins=bins, density=True)
            
            # Calculate KL and JS divergences
            kl_pq, kl_qp, js = calculate_kl_divergence(hist1, hist2)
            
            kl_pq_list.append(kl_pq)
            kl_qp_list.append(kl_qp)
            js_list.append(js)
        
        # Calculate averages across all dimensions
        avg_kl_pq = np.nanmean(kl_pq_list)
        avg_kl_qp = np.nanmean(kl_qp_list)
        avg_js = np.nanmean(js_list)
        
        result = {
            'avg_kl_1_to_2': avg_kl_pq,
            'avg_kl_2_to_1': avg_kl_qp,
            'avg_js': avg_js,
        }
        
        logger.debug(f"Distribution comparison results: {result}")
        return result
    except Exception as e:
        logger.error(f"Error comparing hidden state distributions: {e}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}

def reduce_dimensions(data: np.ndarray, method: str = 'pca', n_components: int = 2) -> np.ndarray:
    """
    Reduce dimensionality of high-dimensional data
    """
    logger.debug(f"Using {method} to reduce {data.shape} data to {n_components} dimensions")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=1000)
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=n_components)
    elif method.lower() == 'zca':
        reducer = zca_whitening(data)
    else:
        logger.warning(f"Unknown reduction method: {method}, using PCA")
        reducer = PCA(n_components=n_components)
    
    reduced_data = reducer.fit_transform(data)
    logger.debug(f"{method} reduction complete, output shape: {reduced_data.shape}")
    return reduced_data

def print_top_generated_content(df_merged, cluster_labels, top_n=3):
    """
    Print top N most common generated content for each cluster.
    """
    df_merged['cluster'] = cluster_labels

    # Iterate through each cluster label
    for cluster in set(cluster_labels):
        if cluster == -1:  # Skip noise points
            continue
        
        cluster_data = df_merged[df_merged['cluster'] == cluster]
        generated_content = cluster_data['generated_content']

        # Count most common content
        most_common = Counter(generated_content).most_common(top_n)

        print(f"\nCluster {cluster}:")
        for content, count in most_common:
            print(f"  {content} (count: {count})")


def zca_whitening(X: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Apply ZCA whitening to data
    """
    logger.debug(f"Performing ZCA whitening on data with shape {X.shape}")
    
    try:
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        U, S, V = np.linalg.svd(cov)
        
        # Compute ZCA transformation matrix
        zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
        
        # Apply ZCA transformation
        whitened = np.dot(X_centered, zca_matrix.T)
        
        logger.debug(f"ZCA whitening complete, output shape: {whitened.shape}")
        return whitened
    except Exception as e:
        logger.error(f"ZCA whitening error: {e}")
        logger.debug(traceback.format_exc())
        exit(1)

def load_word_embeddings_and_labels(
    base_dir: str,
    target_words: List[str],
    extraction_methods: List[str],
    context_window: int,
    data_name: str,
    time_periods: List[str],
    labels_dir: str = "labeled_data_o3_mini"
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    """
    Load word embeddings and label data generated by submit_hs.sh
    """
    logger.info(f"Loading embeddings and labels for {len(target_words)} target words")
    
    try:
        # Initialize result data structures
        embeddings_dict = {}
        combined_dfs = {}
        
        for target_word in target_words:
            logger.debug(f"Processing target word '{target_word}'")
            
            # Convert word format to file path format (replace colon with underscore)
            safe_target_word = target_word.replace(":", "_")
            word, pos = target_word.split(":")
            
            embeddings_dict[word] = {}
            word_dfs = []
            
            for period in time_periods:
                logger.debug(f"Processing period '{period}'")
                period_dfs = []
                
                for method in extraction_methods:
                    # Build hidden state file path
                    hs_dir = Path(base_dir) / safe_target_word / f"{method}_{context_window}" / data_name / period
                    
                    if not hs_dir.exists():
                        logger.warning(f"Directory doesn't exist: {hs_dir}")
                        continue
                    
                    # Find and load .pt files
                    pt_files = list(hs_dir.glob("*.pt"))
                    if not pt_files:
                        logger.warning(f"No .pt files found in {hs_dir}")
                        continue
                    
                    # Load hidden state file
                    try:
                        hidden_states = torch.load(pt_files[0], map_location="cpu")
                        logger.debug(f"Loaded hidden states: {pt_files[0]}, shape: {hidden_states.shape}")
                        
                        # Convert to numpy array and save to word dictionary
                        if isinstance(hidden_states, torch.Tensor):
                            # If multi-layer hidden states, average as word embedding
                            if hidden_states.dim() > 2:
                                embeddings_dict[word][period] = hidden_states.mean(dim=1).numpy()
                            else:
                                embeddings_dict[word][period] = hidden_states.numpy()
                    except Exception as e:
                        logger.error(f"Error loading hidden states: {e}")
                        logger.debug(traceback.format_exc())
                        continue
                    
                    # Find and load CSV files
                    csv_files = list(hs_dir.glob("*.csv"))
                    if not csv_files:
                        logger.warning(f"No CSV files found in {hs_dir}")
                        continue
                    
                    # Load generated CSV file
                    try:
                        df = pd.read_csv(csv_files[0])
                        logger.debug(f"Loaded CSV: {csv_files[0]}, rows: {len(df)}")
                        
                        # Add period and method columns
                        df["time_period"] = period
                        df["extraction_method"] = method
                        
                        period_dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error loading CSV: {e}")
                        logger.debug(traceback.format_exc())
                        continue
                
                if period_dfs:
                    # Merge all DataFrames for current period
                    period_df = pd.concat(period_dfs, ignore_index=True)
                    word_dfs.append(period_df)
            
            if word_dfs:
                # Merge all DataFrames for current word
                word_df = pd.concat(word_dfs, ignore_index=True)
                
                # Load label file
                label_file = Path(labels_dir) / f"{safe_target_word}.csv"
                
                if label_file.exists():
                    try:
                        label_df = pd.read_csv(label_file)
                        logger.debug(f"Loaded label file: {label_file}, rows: {len(label_df)}")
                        
                        # Merge original data and labels
                        merged_df = pd.merge(word_df, label_df, how="inner", on="sentence_id")
                        logger.debug(f"Merged DataFrame rows: {len(merged_df)}")
                        
                        combined_dfs[word] = merged_df
                    except Exception as e:
                        logger.error(f"Error loading or merging label file: {e}")
                        logger.debug(traceback.format_exc())
                        combined_dfs[word] = word_df
                else:
                    logger.warning(f"Label file doesn't exist: {label_file}")
                    combined_dfs[word] = word_df
        
        # Merge all word DataFrames
        if combined_dfs:
            all_words_df = pd.concat(list(combined_dfs.values()), keys=combined_dfs.keys(), names=["word"])
            logger.info(f"Successfully loaded and integrated data, final DataFrame rows: {len(all_words_df)}")
            return embeddings_dict, all_words_df
        else:
            logger.warning("Failed to load any valid data")
            return {}, pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading word embeddings and labels: {e}")
        logger.debug(traceback.format_exc())
        return {}, pd.DataFrame()

def compute_cosine_similarity(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    target_words: List[str],
    time_periods: List[str],
    visualize: bool = True,
    output_dir: Optional[str] = None,
    use_zca: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Calculate cosine similarity between word embeddings across time periods
    """
    logger.info(f"Calculating cosine similarity for {len(target_words)} words across {len(time_periods)} periods")
    
    try:
        results = {}
        
        for word in target_words:
            logger.debug(f"Processing word '{word}'")
            
            # Check if word exists in embeddings dictionary
            if word not in embeddings:
                logger.warning(f"Word '{word}' not in embeddings dictionary, skipping")
                continue
                
            # Check if all periods have embeddings for this word
            missing_periods = [period for period in time_periods if period not in embeddings[word]]
            if missing_periods:
                logger.warning(f"Word '{word}' missing embeddings for periods {', '.join(missing_periods)}, skipping")
                continue
                
            word_similarities = {}
            similarity_values = []
            
            # If using ZCA whitening, preprocess all period embeddings
            if use_zca:
                whitened_embeddings = {}
                # Collect embeddings from all periods
                all_embeddings = np.vstack([embeddings[word][period] for period in time_periods])
                # Apply ZCA whitening to all embeddings together
                all_whitened = zca_whitening(all_embeddings)
                
                # Assign whitened embeddings back to periods
                start_idx = 0
                for i, period in enumerate(time_periods):
                    period_size = embeddings[word][period].shape[0]
                    whitened_embeddings[period] = all_whitened[start_idx:start_idx + period_size]
                    start_idx += period_size
                
                logger.debug(f"Applied ZCA whitening to all embeddings for '{word}'")
            
            # Calculate cosine similarity between adjacent periods
            for i in range(len(time_periods) - 1):
                period1 = time_periods[i]
                period2 = time_periods[i + 1]
                
                # Get embeddings
                if use_zca:
                    vec1 = whitened_embeddings[period1].reshape(1, -1)
                    vec2 = whitened_embeddings[period2].reshape(1, -1)
                else:
                    vec1 = embeddings[word][period1].reshape(1, -1)
                    vec2 = embeddings[word][period2].reshape(1, -1)
                
                # Calculate cosine similarity
                sim = cosine_similarity(vec1, vec2)[0][0]
                period_pair = f"{period1}-{period2}"
                word_similarities[period_pair] = float(sim)
                similarity_values.append(sim)
                
                logger.debug(f"'{word}' similarity for {period_pair}: {sim:.4f}" + 
                           (f" (with ZCA)" if use_zca else ""))
            
            results[word] = word_similarities
            
            # Generate visualization
            if visualize:
                plt.figure(figsize=(10, 6))
                
                # Plot similarity curve
                plt.plot(range(len(similarity_values)), similarity_values, 'o-', linewidth=2, markersize=8)
                
                # Set x-axis labels to period pairs
                period_pairs = [f"{time_periods[i]}-{time_periods[i+1]}" for i in range(len(time_periods)-1)]
                plt.xticks(range(len(period_pairs)), period_pairs, rotation=45)
                
                # Set y-axis range and grid
                plt.ylim(0, 1.05)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add title and labels
                title = f"Semantic change for '{word}' across time periods"
                if use_zca:
                    title += " (with ZCA whitening)"
                plt.title(title)
                plt.ylabel("Cosine Similarity")
                plt.xlabel("Time Periods")
                
                # Add reference lines
                plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3, label='Significant change (0.5)')
                plt.axhline(y=0.8, color='g', linestyle='-', alpha=0.3, label='Minor change (0.8)')
                
                plt.legend()
                plt.tight_layout()
                
                # Save chart
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{word}_semantic_change"
                    if use_zca:
                        filename += "_zca"
                    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
                    logger.debug(f"Saved semantic change chart for '{word}'")
                
                plt.close()
        
        # Generate comparison chart
        if visualize and len(target_words) > 1 and all(word in results for word in target_words):
            plt.figure(figsize=(12, 8))
            
            for word in target_words:
                if word in results:
                    # Extract similarity values
                    similarities = list(results[word].values())
                    # Plot similarity curve for each word
                    plt.plot(range(len(similarities)), similarities, 'o-', linewidth=2, markersize=8, label=word)
            
            # Set x-axis labels to period pairs
            period_pairs = [f"{time_periods[i]}-{time_periods[i+1]}" for i in range(len(time_periods)-1)]
            plt.xticks(range(len(period_pairs)), period_pairs, rotation=45)
            
            # Set y-axis range and grid
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add title and labels
            title = "Multi-word semantic change comparison"
            if use_zca:
                title += " (with ZCA whitening)"
            plt.title(title)
            plt.ylabel("Cosine Similarity")
            plt.xlabel("Time Periods")
            
            # Add reference lines
            plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3, label='Significant change (0.5)')
            plt.axhline(y=0.8, color='g', linestyle='-', alpha=0.3, label='Minor change (0.8)')
            
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            if output_dir:
                filename = "multi_word_semantic_change"
                if use_zca:
                    filename += "_zca"
                plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
                logger.debug("Saved multi-word semantic change comparison chart")
            
            plt.close()
        
        logger.info(f"Cosine similarity calculation completed, processed {len(results)} words")
        return results
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        logger.debug(traceback.format_exc())
        return {}

def merge_label(
    hs_pt_file: str, 
    sentence_csv_file: str, 
    label_csv_file: str = None,
    merge_on: str = "sentence",  # or ["sentence", "year", "target_word", "target_pos"] etc.
    join_type: str = "inner"      # can also be "inner"
) -> pd.DataFrame:
    """
    Merge hidden states file (.pt), sentence file (.csv) and label file (.csv), return DataFrame with labels and hidden states.

    Args:
        hs_pt_file: Path to hidden states file (.pt) (aligned with sentence_csv)
        sentence_csv_file: Path to sentence file (.csv), assumed to contain columns for alignment
        label_csv_file: Path to label file (.csv), example columns: sentence, target_word, target_pos, year, label_index, definition
        merge_on: Column(s) for joining, e.g. "sentence" or ["sentence", "year"]
        join_type: Join method, options: "left", "inner", "outer", "right"

    Returns:
        DataFrame containing:
            - Original sentence info (from sentence_csv_file)
            - Label info (from label_csv_file, NaN for unmatched columns, depends on join_type)
            - hidden_states column (tensors loaded from .pt file, each row corresponds to hidden state of a sentence)
    """
    try:
        # 1. Load hidden states (one-to-one with sentence_csv)
        hidden_states = torch.load(hs_pt_file, map_location="cpu")
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.numpy()
        
        # 2. Load sentence and label files
        df_sentences = pd.read_csv(sentence_csv_file)
        
        # If no label file provided, use sentence data directly
        if label_csv_file is None or not os.path.exists(label_csv_file):
            df_merged = df_sentences.copy()
        else:
            df_labels = pd.read_csv(label_csv_file)
            # 3. Merge sentence and label files
            df_merged = pd.merge(
                df_sentences,
                df_labels,
                on=merge_on,
                how=join_type,
                suffixes=("", "_label")
            )
        
        # 4. Ensure row count matches hidden_states
        if df_merged.shape[0] != hidden_states.shape[0]:
            # Use df_sentences row order as the "gold standard" for alignment
            df_merged = df_merged.loc[df_merged.index.intersection(df_sentences.index)].copy()
            df_merged = df_merged.sort_index()
            
            # If still mismatched, truncate to minimum length
            if df_merged.shape[0] != hidden_states.shape[0]:
                final_len = min(df_merged.shape[0], hidden_states.shape[0])
                df_merged = df_merged.iloc[:final_len].copy()
                hidden_states = hidden_states[:final_len]

        # 5. Store hidden states in DataFrame
        df_merged["hidden_states"] = list(hidden_states)

        print(df_merged.head())
        return df_merged
    except Exception as e:
        print(f"merge_label() error: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

def find_most_rep(generated_contents: List[str]) -> str:
    """
    Find the most representative content (highest frequency) from a list of generated contents.

    Args:
        generated_contents: List of generated contents

    Returns:
        Most common generated content
    """
    if not generated_contents:
        return ""
    
    # Count occurrences of each content
    content_counts = Counter(generated_contents)
    
    # Find most frequent content
    most_common = content_counts.most_common(1)
    if most_common:
        return most_common[0][0]
    return ""

def prepare_hidden_states(hidden_states: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    convert hidden_states to numpy array and ensure a 3D shape [N, L, D].
    """
    hs_np = hidden_states.cpu().numpy()
    # Handle dimensions
    if hs_np.ndim == 4:
        if hs_np.shape[2] > 1:
            # [N, L, B, D] -> [N, L, D] (take first beam)
            logger.debug(f"Input is 4D {hs_np.shape}, selecting first beam.")
            hs_np = hs_np[:, :, 0, :]
        else:
            # [N, L, 1, D] -> [N, L, D]
            hs_np = hs_np[:, :, 0, :]
    else:
        raise ValueError(f"Unsupported hidden_states dimension: {hs_np.ndim}.")
    logger.debug(f"Prepared hidden states shape: {hs_np.shape}")
    return hs_np

def parse_layer_indices(layers_str: str) -> List[int]:
    """
    Parse layer index string, supporting multiple formats.
    
    Format can be:
    1) Comma-separated list: "0,5,12"
    2) Range notation: "0-12"
    3) Combined form: "0,5-9,12"
    
    Args:
        layers_str: Layer index string to parse
        
    Returns:
        Sorted list of layer indices
    """
    if not layers_str:
        return []
    
    indices = set()
    parts = layers_str.split(',')
    
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    
    return sorted(indices)

def parse_year_range(year_str: str) -> Tuple[int, int]:
    """
    Parse year range string, e.g., "1970-1974", into start and end years.
    
    Args:
        year_str: Year range string in "start-end" format
        
    Returns:
        Tuple of start and end years
    """
    if not year_str or '-' not in year_str:
        logger.warning(f"Invalid year range format: {year_str}, should be 'start-end' format")
        return (0, 0)
    
    try:
        start, end = map(int, year_str.split('-'))
        return (start, end)
    except ValueError:
        logger.error(f"Cannot parse year range: {year_str}")
        return (0, 0)

def get_time_periods() -> List[str]:
    """
    Get predefined time periods list for diachronic analysis.
    
    Returns:
        List of predefined time periods in "start-end" format
    """
    return [
        "1970-1974", "1975-1979", "1980-1984", "1985-1989", 
        "1990-1995", "2004-2008", "2009-2013", "2014-2018"
    ]

def filter_by_year(df: pd.DataFrame, start_year: int, end_year: int, year_column: str = 'year') -> pd.DataFrame:
    """
    Filter DataFrame by year range.
    
    Args:
        df: DataFrame to filter
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        year_column: Column name for year in DataFrame, default 'year'
        
    Returns:
        Filtered DataFrame
    """
    if year_column not in df.columns:
        logger.error(f"Year column '{year_column}' not in DataFrame")
        return pd.DataFrame()

    try:
        df[year_column] = pd.to_numeric(df[year_column])
        filtered_df = df[(df[year_column] >= start_year) & (df[year_column] <= end_year)]
        logger.info(f"Year filter results: {len(filtered_df)} of {len(df)} rows kept")
        return filtered_df
    except Exception as e:
        logger.error(f"Year filtering error: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

def resolve_file_path(path, pattern="*.pt", error_message=None, logger=None):
    """
    Resolve file path, supporting direct file path or directory with pattern.
    If directory, uses glob pattern to find matching files.
    
    Args:
        path: File path or directory path
        pattern: Glob pattern for finding files (e.g., "*.pt")
        error_message: Error message when file not found
        logger: Logger instance
    
    Returns:
        Found file path or None
    """
    import os
    import glob
    
    # If path is direct file path and exists, return it
    if os.path.isfile(path):
        if logger:
            logger.info(f"Found file: {path}")
        return path
    
    # If path is directory, find matching files
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, pattern))
        
        if files:
            # Usually return first matching file
            if logger:
                logger.info(f"Found {len(files)} matching files in {path}, using: {files[0]}")
            return files[0]
    
    # If no file found, log error and return None
    if error_message and logger:
        logger.error(error_message)
    elif logger:
        logger.error(f"No matching file found: {path}, pattern: {pattern}")
    
    return None
def cosine_similarity_layers(hidden_states):
    """
    Calculate cosine similarity between different layers of hidden states
    
    Args:
        hidden_states: Hidden states tensor with shape (batch_size, num_layers, 1, hidden_size)
        
    Returns:
        Cosine similarity matrix with shape (batch_size, num_layers, num_layers)
    """
    if hidden_states.ndim != 4:
        raise ValueError("Input hidden states must have shape (batch_size, num_layers, 1, hidden_size)")
    
    if hidden_states.shape[2] != 1:
        hidden_states = hidden_states[:, :, 0, :]
    batch_size, num_layers, _, hidden_size = hidden_states.shape
    res = torch.zeros((batch_size, num_layers, num_layers), device=hidden_states.device)
    
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            vec_i = hidden_states[:, i, 0, :]
            vec_j = hidden_states[:, j, 0, :]
            dot_product = (vec_i * vec_j).sum(dim=-1)
            norm_i = vec_i.norm(dim=-1)
            norm_j = vec_j.norm(dim=-1)
            cos_sim = dot_product / (norm_i * norm_j + 1e-8)
            res[:, i, j] = cos_sim
            res[:, j, i] = cos_sim
    
    return res.cpu().numpy()
def calculate_distribution_entropy(vectors: np.ndarray, n_bins: int = 50) -> float:
    """
    Calculate average entropy of vector distribution to evaluate distribution width/variability.
    
    Higher entropy indicates more dispersed (broad) distribution, lower entropy indicates more concentrated (narrow) distribution.
    
    Args:
        vectors: Vector data with shape [n_samples, n_features]
        n_bins: Number of bins for histogram calculation
        
    Returns:
        Average entropy across all feature dimensions
    """
    try:
        # Ensure input is numpy array
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().numpy()
            
        # Calculate entropy for each feature dimension
        n_features = vectors.shape[1]
        entropies = []
        
        for dim in range(n_features):
            # Extract values for current dimension
            values = vectors[:, dim]
            
            # Calculate histogram to get distribution
            hist, _ = np.histogram(values, bins=n_bins, density=True)
            
            # Add small constant to prevent log(0)
            hist = hist + 1e-10
            
            # Normalize to ensure probability distribution
            hist = hist / np.sum(hist)
            
            # Calculate entropy using scipy's entropy function
            dim_entropy = entropy(hist)
            entropies.append(dim_entropy)
        
        # Calculate average entropy
        avg_entropy = np.mean(entropies)
        
        logger.debug(f"Average distribution entropy: {avg_entropy}")
        return avg_entropy
    except Exception as e:
        logger.error(f"Error calculating distribution entropy: {e}")
        logger.debug(traceback.format_exc())
        return float('nan')