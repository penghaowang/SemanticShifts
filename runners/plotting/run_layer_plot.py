import argparse
import os
import glob
import torch
import numpy as np
from evaluate import cosine_similarity_layers, resolve_file_path
from plot import plot_heatmap_layerwise, plot_relative_similarity
from logger_config import setup_logger



# Set up module-specific logger
logger = setup_logger('run_layer_plot', 'logs/run_layer_plot.log')

# First, make these two plots for each word, then make these two plots for the whole set

def main():
    parser = argparse.ArgumentParser(description='Layer-wise plot')
    parser.add_argument('--base_hs_dir', type=str, required=True, help='Base directory containing all hidden states')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--method', type=str, required=True, help='Method name (e.g., input_last_token, eos_token)')
    args = parser.parse_args()

    logger.info(f"Starting inter-layer similarity analysis, Base directory: {args.base_hs_dir}, Method: {args.method}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory created: {args.output_dir}")

    # Find all hidden state directories containing this method
    word_dirs_pattern = os.path.join(args.base_hs_dir, "*", args.method, "combined", "hidden_states_*")
    word_dirs = glob.glob(word_dirs_pattern)
    
    if not word_dirs:
        logger.error(f"No matching hidden state directories found: {word_dirs_pattern}")
        return
    
    logger.info(f"Found {len(word_dirs)} vocabulary directories")
    
    # Create a dictionary to store layer similarity for each word
    word_layer_similarities = {}
    all_layer_similarities = []
    
    # Iterate through each vocabulary directory
    for word_dir in word_dirs:
        # Extract vocabulary name from directory path
        word_folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(word_dir))))
        logger.info(f"Processing vocabulary directory: {word_folder}")
        
        # Find all hidden state files in this directory
        hidden_states_files = glob.glob(os.path.join(word_dir, "*.pt"))
        
        if not hidden_states_files:
            logger.warning(f"No hidden state files found in vocabulary directory '{word_folder}', skipping")
            continue
        
        logger.info(f"Found {len(hidden_states_files)} files for vocabulary {word_folder}")
        
        # Iterate through all hidden state files for this vocabulary
        for file in hidden_states_files:
            logger.info(f"Processing file: {file}")
            
            # Load hidden states
            try:
                hidden_states = torch.load(file)
                logger.debug(f"Loaded hidden states, shape: {hidden_states.shape}")
                
                # Calculate layer-wise similarity
                logger.info(f"Calculating inter-layer similarity for {word_folder}")
                layer_similarity = cosine_similarity_layers(hidden_states)
                
                # Store layer similarity for this word
                word_key = f"{word_folder}_{os.path.basename(file)}"
                word_layer_similarities[word_key] = layer_similarity.cpu().numpy() if isinstance(layer_similarity, torch.Tensor) else layer_similarity
                
                # Add to the total layer similarity list
                all_layer_similarities.append(layer_similarity)
                
                # Plot layer-wise heatmap
                heatmap_path = os.path.join(args.output_dir, f"{word_folder}_{os.path.basename(file)}_heatmap.png")
                logger.info(f"Plotting layer-wise heatmap for {word_folder}, saving to: {heatmap_path}")
                plot_heatmap_layerwise(layer_similarity, heatmap_path)
                
                # Plot relative similarity heatmap
                rel_similarity_path = os.path.join(args.output_dir, f"{word_folder}_{os.path.basename(file)}_rel_similarity.png")
                logger.info(f"Plotting relative similarity plot for {word_folder}, saving to: {rel_similarity_path}")
                plot_relative_similarity(layer_similarity, rel_similarity_path)
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
    
    # If there are successfully processed vocabularies, plot the overall chart
    if all_layer_similarities:
        logger.info("Starting to plot the overall inter-layer similarity chart for all vocabularies")
        
        # Merge all layer similarities
        try:
            all_layer_similarity = np.concatenate([sim.cpu().numpy() if isinstance(sim, torch.Tensor) else sim for sim in all_layer_similarities], axis=0)
            logger.debug(f"Shape of merged inter-layer similarity array: {all_layer_similarity.shape}")
            
            # Plot layer-wise heatmap
            heatmap_path = os.path.join(args.output_dir, "all_words_layer_heatmap.png")
            logger.info(f"Plotting layer-wise heatmap for all vocabularies, saving to: {heatmap_path}")
            plot_heatmap_layerwise(all_layer_similarity, heatmap_path)
            
            # Plot relative similarity heatmap
            rel_similarity_path = os.path.join(args.output_dir, "all_words_rel_similarity.png")
            logger.info(f"Plotting relative similarity plot for all vocabularies, saving to: {rel_similarity_path}")
            plot_relative_similarity(all_layer_similarity, rel_similarity_path)
        except Exception as e:
            logger.error(f"Failed to merge layer similarity arrays: {e}")
    else:
        logger.warning("No vocabularies processed successfully, cannot plot overall chart")
    
    logger.info("Inter-layer similarity analysis completed")

if __name__ == "__main__":
    main()
