import os
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity
import logging
import glob
import torch
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from plot import plot_correlation_heatmap

# Set logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extract_correlation.log')
    ]
)
logger = logging.getLogger('extract_correlation')

def load_sample(file_path: str) -> Union[np.ndarray, None]:
    """Load a single sample file, support .npy and .pt formats, and unify the shape"""
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.pt'):
            data = torch.load(file_path, map_location='cpu').cpu().numpy()
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return None
            
        # Unify data shape: if the third dimension is greater than 1, take the first element
        if data.ndim == 4 and data.shape[2] > 1:
            logger.info(f"Detected data with third dimension > 1: {data.shape}, taking the first element of the third dimension")
            # This maintains the four-dimensional structure
            data = data[:, :, 0:1, :]  # Maintain 4D structure
            logger.info(f"Shape after processing: {data.shape}")
            
        return data
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None

def load_method_samples(method_folder: str, file_pattern: str = "*.pt") -> Union[np.ndarray, None]:
    """Load all sample files from the given method folder"""
    file_paths = glob.glob(os.path.join(method_folder, file_pattern))
    
    if not file_paths:
        logger.warning(f"No files matching {file_pattern} found in {method_folder}")
        return None
    
    # If there is only one file, load it directly
    if len(file_paths) == 1:
        logger.info(f"Found a single file in {method_folder}, loading directly")
        return load_sample(file_paths[0])
    
    # Use multiprocessing to load multiple files
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(load_sample, file_paths), 
                          total=len(file_paths),
                          desc=f"Loading samples from {os.path.basename(method_folder)}"))
    
    # Filter out failed samples and merge
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        logger.warning(f"No samples were successfully loaded from {method_folder}")
        return None
    
    try:
        return np.concatenate(valid_results, axis=0)
    except ValueError as e:
        logger.error(f"Failed to merge samples: {e}")
        return None

def compute_cosine_similarity_with_layers(v1: np.ndarray, v2: np.ndarray, layer_indices: List[int]) -> float:
    """Compute cosine similarity between two samples on specified layers"""
    # Ensure input is a 4D tensor [batch=1, layers, 1, hid_dim]
    assert v1.ndim == 4 and v2.ndim == 4, "Input tensor must be 4-dimensional"
    assert v1.shape[0] == 1 and v2.shape[0] == 1, "Batch size must be 1"
    
    n_layers = v1.shape[1]
    
    # Ensure layer indices are valid
    valid_indices = [l for l in layer_indices if 0 <= l < n_layers]
    if not valid_indices:
        logger.warning(f"All provided layer indices {layer_indices} are out of range [0, {n_layers-1}]")
        return 0.0
    
    # Remove the third dimension (value is 1) and select specific layers
    similarities = []
    for l in valid_indices:
        v1_vec = v1[0, l].reshape(1, -1)
        v2_vec = v2[0, l].reshape(1, -1)
        sim = cosine_similarity(v1_vec, v2_vec)[0][0]
        similarities.append(sim)
    
    # Return the average similarity of all selected layers
    return np.mean(similarities)

def compute_batch_correlation(data_i: np.ndarray, data_j: np.ndarray, 
                             batch_start: int, batch_end: int, 
                             layer_indices: List[int]) -> np.ndarray:
    """Compute correlation for a batch of samples"""
    batch_data_i = data_i[batch_start:batch_end]
    batch_data_j = data_j[batch_start:batch_end]
    
    batch_size = batch_data_i.shape[0]
    batch_similarity = np.zeros(batch_size)
    
    for k in range(batch_size):
        v_i = batch_data_i[k:k+1]  # Maintain 4D shape [1, layers, 1, hid_dim]
        v_j = batch_data_j[k:k+1]  # Maintain 4D shape [1, layers, 1, hid_dim]
        batch_similarity[k] = compute_cosine_similarity_with_layers(v_i, v_j, layer_indices)
    
    return batch_similarity

def parallel_batch_compute_correlation(method_data: List[np.ndarray], 
                                      batch_size: int, 
                                      layer_indices: List[int]) -> np.ndarray:
    """Compute the correlation matrix between multiple methods in parallel"""
    num_methods = len(method_data)
    num_samples = min(m.shape[0] for m in method_data)  # Use the minimum number of samples among all methods
    
    # Record the shape information for each method
    for i, data in enumerate(method_data):
        logger.info(f"Method {i+1} data shape: {data.shape}")
    
    logger.info(f"Computing correlation using {num_samples} samples and layers {layer_indices}")
    
    # Truncate all method data to the same number of samples
    method_data = [m[:num_samples] for m in method_data]
    
    # Initialize correlation matrix and set diagonal to 1.0
    correlations = np.zeros((num_methods, num_methods))
    for i in range(num_methods):
        correlations[i, i] = 1.0
    
    # Create a multiprocessing pool to compute correlation
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i in range(num_methods):
            for j in range(i+1, num_methods):
                data_i = method_data[i]
                data_j = method_data[j]
                
                logger.info(f"Computing correlation between method {i+1} and method {j+1}")
                
                # Create batch tasks
                results = []
                for batch_start in range(0, num_samples, batch_size):
                    batch_end = min(batch_start + batch_size, num_samples)
                    results.append(
                        pool.apply_async(
                            compute_batch_correlation,
                            (data_i, data_j, batch_start, batch_end, layer_indices)
                        )
                    )
                
                # Collect results and compute average similarity
                batch_similarities = [result.get() for result in results]
                all_similarities = np.concatenate(batch_similarities)
                avg_similarity = np.mean(all_similarities)
                
                # Fill the correlation matrix (symmetric matrix)
                correlations[i, j] = avg_similarity
                correlations[j, i] = avg_similarity
                
                logger.info(f"Correlation between method {i+1} and method {j+1}: {avg_similarity:.4f}")
    
    return correlations

def process_word_folder(word_folder: str, base_hs_dir: str, methods: List[str], 
                       batch_size: int, file_pattern: str, output_dir: str,
                       layer_indices: List[int]) -> Tuple[np.ndarray, List[str]]:
    """Process a single word folder"""
    logger.info(f"======= Processing word: {word_folder} =======")
    
    # Collect method paths and names
    method_paths = []
    method_names = []
    
    for method in methods:
        # Find the latest hidden state directory for this method
        hs_dir_pattern = os.path.join(base_hs_dir, word_folder, method, "combined", "hidden_states_*")
        hs_dirs = glob.glob(hs_dir_pattern)
        
        if not hs_dirs:
            logger.warning(f"Hidden state directory not found for method {method}")
            continue
        
        # Select the latest directory
        hs_dir = max(hs_dirs, key=os.path.getmtime)
        method_paths.append(hs_dir)
        method_names.append(method)
    
    # Ensure there are at least two methods
    if len(method_paths) < 2:
        logger.warning(f"Word {word_folder} has less than 2 valid methods, skipping")
        return None, None
    
    # Load data for each method
    method_data = []
    valid_method_names = []
    
    for folder, name in zip(method_paths, method_names):
        data = load_method_samples(folder, file_pattern)
        if data is not None and data.ndim == 4:
            method_data.append(data)
            valid_method_names.append(name)
            logger.info(f"Loaded data for {name}: {data.shape[0]} samples, {data.shape[1]} layers, third dimension size: {data.shape[2]}")
        else:
            if data is not None:
                logger.error(f"Data shape for method {name} is incorrect: {data.shape}")
            else:
                logger.error(f"Unable to load data for method {name}")
    
    # Ensure there are at least two valid method data
    if len(method_data) < 2:
        logger.warning(f"Word {word_folder} requires at least two valid method data, skipping")
        return None, None
    
    # Create output directory
    word_output_dir = os.path.join(output_dir, word_folder)
    os.makedirs(word_output_dir, exist_ok=True)
    
    # Compute correlation matrix
    logger.info(f"Computing method correlation for word {word_folder}")
    correlation_matrix = parallel_batch_compute_correlation(method_data, batch_size, layer_indices)
    
    # Plot and save correlation heatmap
    output_file = os.path.join(word_output_dir, f"{word_folder}_method_correlation.png")
    plot_correlation_heatmap(
        correlation_matrix=correlation_matrix, 
        method_names=valid_method_names,
        title=f"Method Correlation for {word_folder}",
        output_file=output_file
    )
    
    return correlation_matrix, valid_method_names

def aggregate_correlations(all_correlations: List[np.ndarray], method_names: List[str], 
                         output_file: str, output_dir: str, layer_indices: List[int]):
    """Aggregate correlation matrices for all words"""
    logger.info("Aggregating correlation matrices for all words")
    
    if not all_correlations:
        logger.error("No valid correlation matrices to aggregate")
        return
    
    # Compute average correlation matrix
    avg_correlation = np.mean(all_correlations, axis=0)
    
    # Save aggregated correlation matrix
    np.save(os.path.join(output_dir, "aggregate_correlation_matrix.npy"), avg_correlation)
    
    # Plot and save aggregated heatmap
    layers_str = ",".join(map(str, layer_indices))
    plot_correlation_heatmap(
        correlation_matrix=avg_correlation,
        method_names=method_names,
        title=f"Aggregate Method Correlation (Layers: {layers_str})",
        output_file=output_file
    )

def main():
    parser = argparse.ArgumentParser(description="Compute correlation between different extraction methods")
    parser.add_argument("--base_dir", required=True, help="Base hidden state directory")
    parser.add_argument("--word_folders", required=True, help="Word folders to process, comma-separated")
    parser.add_argument("--methods", required=True, help="Extraction methods to compare, comma-separated")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for computing correlation")
    parser.add_argument("--file_pattern", default="*.pt", help="Hidden state file pattern")
    parser.add_argument("--layers", default="1", help="Layer indices to use, comma-separated (0-based)")
    
    args = parser.parse_args()
    
    # Parse layer indices
    layer_indices = [int(idx) for idx in args.layers.split(",")]
    
    # Parse word folders and methods
    word_folders = args.word_folders.split(",")
    methods = args.methods.split(",")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each word folder
    all_correlations = []
    common_method_names = None
    
    for folder in word_folders:
        correlation_matrix, method_names = process_word_folder(
            folder, args.base_dir, methods, args.batch_size, 
            args.file_pattern, args.output_dir, layer_indices
        )
        
        if correlation_matrix is not None:
            all_correlations.append(correlation_matrix)
            
            # Save the first list of valid method names as common method names
            if common_method_names is None:
                common_method_names = method_names
    
    # If there are valid correlation matrices, aggregate them
    if all_correlations and common_method_names:
        aggregate_output_file = os.path.join(args.output_dir, "aggregate_method_correlation.png")
        aggregate_correlations(
            all_correlations, common_method_names, 
            aggregate_output_file, args.output_dir, layer_indices
        )
    else:
        logger.error("No valid correlation matrices to aggregate")

if __name__ == "__main__":
    main() 