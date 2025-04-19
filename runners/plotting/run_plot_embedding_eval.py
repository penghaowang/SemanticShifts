"""
Word Embedding Evaluation Module

This module contains functions to evaluate the quality of hidden states as word embeddings.
It includes tests for:
1. Semantic similarity consistency
2. Word analogy testing
3. Cluster coherence
4. Dimensional reduction visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import os
import traceback
from pathlib import Path

# 导入自定义日志配置
from logger_config import setup_logger, TRACE

# 设置模块特定的日志记录器
logger = setup_logger('embedding_eval', 'logs/embedding_eval.log')

class EmbeddingEvaluator:
    """Evaluate word embeddings quality"""
    
    def __init__(self, 
                 hidden_states: torch.Tensor,
                 word_info: Optional[pd.DataFrame] = None,
                 output_dir: str = "embedding_evaluation",
                 log_file: str = "logs/embedding_eval.log"):
        """
        Initialize the evaluator with hidden states and word information.
        
        Args:
            hidden_states: Tensor of shape [samples, ...] containing word vectors
            word_info: DataFrame with information about each sample
            output_dir: Directory to save evaluation results
            log_file: Path to log file
        """
        # 如果指定了自定义日志文件，重新配置日志记录器
        if log_file != 'logs/embedding_eval.log':
            global logger
            logger = setup_logger('embedding_eval', log_file)
            
        self.hidden_states = hidden_states
        self.word_info = word_info
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare vectors based on hidden states shape
        if hidden_states.ndim == 3:  # [samples, layers, dim]
            logger.info(f"隐藏状态形状为 {hidden_states.shape} - 将评估每一层")
            self.n_layers = hidden_states.shape[1]
            self.word_vectors = [
                hidden_states[:, layer_idx, :].cpu().numpy() 
                for layer_idx in range(self.n_layers)
            ]
        else:  # [samples, dim]
            logger.info(f"隐藏状态形状为 {hidden_states.shape} - 单层评估")
            self.n_layers = 1
            self.word_vectors = [hidden_states.cpu().numpy()]
    
    def evaluate_all(self):
        """Run all evaluation methods and generate a report"""
        report = {}
        
        # Run evaluations per layer
        for layer_idx in range(self.n_layers):
            layer_name = f"layer_{layer_idx}" if self.n_layers > 1 else "all"
            vectors = self.word_vectors[layer_idx]
            
            # Get sample words if word_info is available
            sample_words = None
            if self.word_info is not None:
                # Extract sample text from word_info
                sample_words = self.word_info['sentence'].tolist()
            
            logger.info(f"Evaluating {layer_name}...")
            
            layer_report = {
                # Lexical similarity measures
                "similarity_stats": self.evaluate_similarity_consistency(
                    vectors, layer_name, sample_words
                ),
                
                # Cluster analysis
                "cluster_quality": self.evaluate_cluster_quality(
                    vectors, layer_name, sample_words
                ),
                
                # Vector space properties
                "vector_properties": self.evaluate_vector_properties(
                    vectors, layer_name
                ),
                
                # Nearest neighbors analysis
                "nearest_neighbors": self.evaluate_nearest_neighbors(
                    vectors, layer_name, sample_words, top_k=5
                )
            }
            
            report[layer_name] = layer_report
        
        # Save report
        report_path = os.path.join(self.output_dir, "embedding_quality_report.txt")
        with open(report_path, 'w') as f:
            self._write_report(f, report)
        
        logger.info(f"Evaluation complete. Report saved to {report_path}")
        return report
    
    def evaluate_similarity_consistency(self, 
                                        vectors: np.ndarray, 
                                        layer_name: str,
                                        sample_words: Optional[List[str]] = None) -> Dict:
        """
        Evaluate if similar words have similar vectors.
        
        Args:
            vectors: Word vectors of shape [samples, dim]
            layer_name: Name of the layer for output files
            sample_words: Optional list of words for the vectors
        
        Returns:
            Dictionary with similarity statistics
        """
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(vectors)
        
        # Calculate similarity statistics
        sim_stats = {
            "mean_similarity": np.mean(sim_matrix),
            "median_similarity": np.median(sim_matrix),
            "min_similarity": np.min(sim_matrix),
            "max_similarity": np.max(sim_matrix),
            "std_similarity": np.std(sim_matrix)
        }
        
        # Plot similarity matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, cmap='viridis')
        plt.title(f"{layer_name} - Cosine Similarity Matrix")
        
        # Save plot
        plt.savefig(
            os.path.join(self.output_dir, f"{layer_name}_similarity_matrix.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # If we have sample words, save top similar word pairs
        if sample_words and len(sample_words) == len(vectors):
            # Get top similar pairs (excluding self-similarity)
            np.fill_diagonal(sim_matrix, -1)  # Replace self-similarity with -1
            n_pairs = min(20, len(vectors))
            
            # Find top similar pairs
            top_pairs = []
            for _ in range(n_pairs):
                max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
                word1, word2 = sample_words[max_idx[0]], sample_words[max_idx[1]]
                similarity = sim_matrix[max_idx]
                top_pairs.append((word1, word2, similarity))
                sim_matrix[max_idx] = -1  # Mark as processed
            
            # Add to report
            sim_stats["top_similar_pairs"] = top_pairs
        
        return sim_stats
    
    def evaluate_cluster_quality(self,
                                vectors: np.ndarray,
                                layer_name: str,
                                sample_words: Optional[List[str]] = None) -> Dict:
        """
        Evaluate if the vectors form meaningful clusters.
        
        Args:
            vectors: Word vectors of shape [samples, dim]
            layer_name: Name of the layer for output files
            sample_words: Optional list of words for the vectors
        
        Returns:
            Dictionary with cluster quality metrics
        """
        # Determine number of clusters based on data size
        n_clusters = min(max(3, len(vectors) // 10), 10)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(vectors)
        
        # Calculate silhouette score (measure of cluster quality)
        try:
            silhouette = silhouette_score(vectors, clusters)
        except:
            silhouette = 0.0  # Default if calculation fails
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        # Calculate cluster sizes
        cluster_sizes = np.bincount(clusters)
        
        # Calculate average distances to centroids
        distances = []
        for i, vector in enumerate(vectors):
            cluster_idx = clusters[i]
            centroid = kmeans.cluster_centers_[cluster_idx]
            distance = np.linalg.norm(vector - centroid)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # If we have sample words, analyze word clusters
        word_clusters = None
        if sample_words and len(sample_words) == len(vectors):
            word_clusters = {}
            for cluster_idx in range(n_clusters):
                # Get indices of words in this cluster
                cluster_indices = np.where(clusters == cluster_idx)[0]
                # Get the words
                cluster_words = [sample_words[i] for i in cluster_indices]
                word_clusters[f"cluster_{cluster_idx}"] = cluster_words
        
        # Create cluster quality metrics
        cluster_metrics = {
            "silhouette_score": silhouette,
            "inertia": inertia,
            "num_clusters": n_clusters,
            "cluster_sizes": cluster_sizes.tolist(),
            "avg_distance_to_centroid": avg_distance,
            "word_clusters": word_clusters
        }
        
        # Plot clusters if 2D vectors are available
        # For high-dimensional vectors, we'll reduce to 2D for visualization
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        plt.figure(figsize=(12, 10))
        
        # Plot each cluster with a different color
        for cluster_idx in range(n_clusters):
            indices = np.where(clusters == cluster_idx)[0]
            plt.scatter(
                vectors_2d[indices, 0],
                vectors_2d[indices, 1],
                label=f"Cluster {cluster_idx} (n={len(indices)})",
                alpha=0.7
            )
        
        # Add cluster centroids
        centroids_2d = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            s=100,
            c='black',
            marker='x',
            label='Centroids'
        )
        
        plt.title(f"{layer_name} - Word Clusters (Silhouette: {silhouette:.3f})")
        plt.legend()
        
        # Save plot
        plt.savefig(
            os.path.join(self.output_dir, f"{layer_name}_clusters.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        return cluster_metrics
    
    def evaluate_vector_properties(self,
                                  vectors: np.ndarray,
                                  layer_name: str) -> Dict:
        """
        Evaluate statistical properties of the vector space.
        
        Args:
            vectors: Word vectors of shape [samples, dim]
            layer_name: Name of the layer for output files
        
        Returns:
            Dictionary with vector space properties
        """
        # Calculate basic statistics
        mean_vector = np.mean(vectors, axis=0)
        std_vector = np.std(vectors, axis=0)
        
        # Calculate vector norms
        norms = np.linalg.norm(vectors, axis=1)
        
        # Calculate dimensionality statistics
        effective_dim = np.linalg.matrix_rank(vectors)
        
        # Calculate principal components to analyze variance distribution
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(vectors)
        
        # Calculate explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        
        # Calculate how many components explain 90% of variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        components_90pct = np.argmax(cumulative_variance >= 0.9) + 1
        
        # Plot PCA explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(explained_variance_ratio), marker='o', linestyle='-', markersize=4)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f"{layer_name} - PCA Explained Variance")
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at 90%
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
        plt.axvline(x=components_90pct, color='g', linestyle='--', 
                   label=f'Components for 90%: {components_90pct}')
        plt.legend()
        
        # Save plot
        plt.savefig(
            os.path.join(self.output_dir, f"{layer_name}_pca_variance.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Plot vector norms distribution
        plt.figure(figsize=(10, 6))
        plt.hist(norms, bins=30, alpha=0.7)
        plt.axvline(x=np.mean(norms), color='r', linestyle='--', label=f'Mean: {np.mean(norms):.2f}')
        plt.xlabel('Vector Norm')
        plt.ylabel('Frequency')
        plt.title(f"{layer_name} - Vector Norm Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(
            os.path.join(self.output_dir, f"{layer_name}_norm_distribution.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Return properties
        return {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "effective_dimension": int(effective_dim),
            "components_for_90pct_variance": int(components_90pct),
            "input_dims": vectors.shape[1],
            "total_vectors": vectors.shape[0]
        }
    
    def evaluate_nearest_neighbors(self,
                                  vectors: np.ndarray,
                                  layer_name: str,
                                  sample_words: Optional[List[str]] = None,
                                  top_k: int = 5) -> Dict:
        """
        Evaluate nearest neighbors for each vector.
        
        Args:
            vectors: Word vectors of shape [samples, dim]
            layer_name: Name of the layer for output files
            sample_words: Optional list of words for the vectors
            top_k: Number of nearest neighbors to find
        
        Returns:
            Dictionary with nearest neighbor information
        """
        # Create a nearest neighbors model
        n_neighbors = min(top_k + 1, len(vectors))  # +1 because the word itself is included
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(vectors)
        
        # Find nearest neighbors for each vector
        distances, indices = nbrs.kneighbors(vectors)
        
        # Skip the first column (self)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Calculate average distance to nearest neighbors
        avg_distance = np.mean(distances)
        
        # If we have sample words, get word neighbors
        word_neighbors = None
        if sample_words and len(sample_words) == len(vectors):
            word_neighbors = {}
            
            # Select a subset of examples to display
            sample_indices = np.linspace(0, len(vectors)-1, min(10, len(vectors)), dtype=int)
            
            for sample_idx in sample_indices:
                word = sample_words[sample_idx]
                neighbors = []
                
                for i, neighbor_idx in enumerate(indices[sample_idx]):
                    neighbor_word = sample_words[neighbor_idx]
                    neighbor_distance = distances[sample_idx][i]
                    neighbors.append((neighbor_word, float(neighbor_distance)))
                
                word_neighbors[word] = neighbors
        
        # Return nearest neighbor data
        return {
            "avg_distance_to_neighbors": float(avg_distance),
            "word_neighbors": word_neighbors
        }
    
    def _write_report(self, file_obj, report):
        """Write the evaluation report to a file"""
        file_obj.write("=== Word Embedding Evaluation Report ===\n\n")
        
        for layer_name, layer_report in report.items():
            file_obj.write(f"### {layer_name.upper()} ###\n\n")
            
            # Write similarity statistics
            file_obj.write("-- Similarity Statistics --\n")
            sim_stats = layer_report["similarity_stats"]
            for stat_name, stat_value in sim_stats.items():
                if stat_name != "top_similar_pairs":
                    file_obj.write(f"{stat_name}: {stat_value:.4f}\n")
            
            # Write top similar pairs if available
            if "top_similar_pairs" in sim_stats:
                file_obj.write("\nTop Similar Word Pairs:\n")
                for i, (word1, word2, sim) in enumerate(sim_stats["top_similar_pairs"]):
                    # Truncate long words
                    word1_short = (word1[:20] + '...') if len(word1) > 23 else word1
                    word2_short = (word2[:20] + '...') if len(word2) > 23 else word2
                    file_obj.write(f"{i+1}. \"{word1_short}\" and \"{word2_short}\": {sim:.4f}\n")
            
            file_obj.write("\n")
            
            # Write cluster quality metrics
            file_obj.write("-- Cluster Quality --\n")
            cluster_metrics = layer_report["cluster_quality"]
            file_obj.write(f"Silhouette Score: {cluster_metrics['silhouette_score']:.4f}\n")
            file_obj.write(f"Inertia: {cluster_metrics['inertia']:.4f}\n")
            file_obj.write(f"Number of Clusters: {cluster_metrics['num_clusters']}\n")
            file_obj.write(f"Average Distance to Centroid: {cluster_metrics['avg_distance_to_centroid']:.4f}\n")
            file_obj.write(f"Cluster Sizes: {cluster_metrics['cluster_sizes']}\n")
            
            # Write cluster words if available
            if cluster_metrics["word_clusters"]:
                file_obj.write("\nWord Clusters (sample):\n")
                for cluster_name, words in cluster_metrics["word_clusters"].items():
                    # Limit to 5 examples per cluster
                    sample_words = words[:5]
                    words_str = ", ".join(f'"{w}"' for w in sample_words)
                    if len(words) > 5:
                        words_str += f" and {len(words)-5} more"
                    file_obj.write(f"{cluster_name}: {words_str}\n")
            
            file_obj.write("\n")
            
            # Write vector properties
            file_obj.write("-- Vector Space Properties --\n")
            vector_props = layer_report["vector_properties"]
            file_obj.write(f"Vector Dimensions: {vector_props['input_dims']}\n")
            file_obj.write(f"Total Vectors: {vector_props['total_vectors']}\n")
            file_obj.write(f"Effective Dimension: {vector_props['effective_dimension']}\n")
            file_obj.write(f"Components for 90% Variance: {vector_props['components_for_90pct_variance']}\n")
            file_obj.write(f"Mean Vector Norm: {vector_props['mean_norm']:.4f}\n")
            file_obj.write(f"Std Vector Norm: {vector_props['std_norm']:.4f}\n")
            file_obj.write(f"Min/Max Vector Norm: {vector_props['min_norm']:.4f} / {vector_props['max_norm']:.4f}\n")
            
            file_obj.write("\n")
            
            # Write nearest neighbor information
            file_obj.write("-- Nearest Neighbors --\n")
            nn_data = layer_report["nearest_neighbors"]
            file_obj.write(f"Average Distance to Neighbors: {nn_data['avg_distance_to_neighbors']:.4f}\n")
            
            # Write neighbor examples if available
            if nn_data["word_neighbors"]:
                file_obj.write("\nSample Nearest Neighbors:\n")
                for word, neighbors in nn_data["word_neighbors"].items():
                    # Truncate long word
                    word_short = (word[:20] + '...') if len(word) > 23 else word
                    file_obj.write(f'"{word_short}" → ')
                    neighbor_str = ", ".join(f'"{n[0]}" ({n[1]:.3f})' for n in neighbors[:3])
                    file_obj.write(f"{neighbor_str}\n")
            
            file_obj.write("\n" + "="*50 + "\n\n")

# Usage example
def evaluate_embeddings(hidden_states_path, word_data_path=None, output_dir="embedding_eval_results"):
    """Load and evaluate embeddings from file"""
    # Load hidden states
    hidden_states = torch.load(hidden_states_path, map_location=torch.device('cpu'))
    logger.info(f"Loaded hidden states with shape {hidden_states.shape}")
    
    # Load word data if available
    word_info = None
    if word_data_path and os.path.exists(word_data_path):
        try:
            word_info = pd.read_csv(word_data_path)
            logger.info(f"Loaded word info with {len(word_info)} entries")
        except Exception as e:
            logger.error(f"Error loading word data: {e}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    evaluator = EmbeddingEvaluator(hidden_states, word_info, output_dir)
    report = evaluator.evaluate_all()
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate word embeddings quality")
    parser.add_argument('--hidden_states_path', type=str, required=True,
                        help='Path to the hidden states tensor file')
    parser.add_argument('--word_data_path', type=str, default=None,
                        help='Path to CSV file with word information (optional)')
    parser.add_argument('--output_dir', type=str, default="embedding_eval_results",
                        help='Directory to save evaluation outputs')
    
    args = parser.parse_args()
    
    evaluate_embeddings(
        args.hidden_states_path,
        args.word_data_path,
        args.output_dir
    )