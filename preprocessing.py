import pandas as pd
import torch
import spacy
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import os
import logging
import time

class TextPreprocessor:
    def __init__(self, 
                 accelerator: Optional[Accelerator] = None,
                 spacy_model: str = 'en_core_web_sm',
                 max_batch_size: int = 1000,
                 num_workers: Optional[int] = None):
        """
        Enhanced initialization with memory management and multi-GPU support
        
        Args:
            accelerator: Optional Accelerator instance for multi-GPU support
            spacy_model: Name of spacy model to load
            max_batch_size: Maximum batch size for processing
            num_workers: Number of workers for parallel processing. Defaults to CPU count.
        """
        self.accelerator = accelerator or Accelerator(
            mixed_precision='fp16',
            gradient_accumulation_steps=1,
            device_placement=True,
            split_batches=True
        )
        self.device = self.accelerator.device
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spacy model {spacy_model}")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers or mp.cpu_count()

    @staticmethod
    def estimate_batch_size(available_memory: int, 
                          model_size: int, 
                          tokenizer_overhead: int = 1024) -> int:
        """Estimate optimal batch size based on available memory"""
        per_sample_memory = model_size + tokenizer_overhead
        return min(32, max(1, available_memory // per_sample_memory))

    def fix_text_errors(self, text: str) -> str:
        """Fix common OCR/PDF conversion errors in text."""
        error_fixes = {
            'Þ': 'fi',
            # Add more error fixes as needed
        }
        
        for error, fix in error_fixes.items():
            text = text.replace(error, fix)
        return text

    def calculate_perplexity_batch(self,
                                 sentences: List[str],
                                 model: Any,
                                 tokenizer: Any,
                                 max_length: int = 256) -> List[float]:
        """
        Calculate perplexity scores for a batch of sentences using multiple GPUs
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for distributed setup
        model = self.accelerator.prepare(model)
        all_perplexities = []
        
        # Adjust batch size for multi-GPU setup
        effective_batch_size = self.max_batch_size // self.accelerator.num_processes
        
        for i in range(0, len(sentences), effective_batch_size):
            batch_sentences = sentences[i:i + effective_batch_size]

            batch_inputs = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            # Prepare inputs for distributed processing
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            batch_inputs = self.accelerator.prepare(batch_inputs)

            with torch.no_grad():
                outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
                logits = outputs.logits
                
                # Gather results from all GPUs
                if self.accelerator.num_processes > 1:
                    logits = self.accelerator.gather(logits)
                    batch_inputs = {k: self.accelerator.gather(v) for k, v in batch_inputs.items()}
                
                # Calculate perplexities as before
                batch_perplexities = []
                for idx in range(len(batch_sentences)):
                    mask = batch_inputs['attention_mask'][idx]
                    seq_length = mask.sum().item()
                    seq_logits = logits[idx][:seq_length]
                    seq_labels = batch_inputs['input_ids'][idx][:seq_length]
                    seq_loss = torch.nn.functional.cross_entropy(
                        seq_logits[:-1], 
                        seq_labels[1:],
                        reduction='mean'
                    )
                    seq_perplexity = torch.exp(seq_loss).item()
                    batch_perplexities.append(seq_perplexity)
                
                all_perplexities.extend(batch_perplexities)
                    
        return all_perplexities

    def process_batch(self, sentences: List[str], 
                     model: Optional[Any] = None,
                     tokenizer: Optional[Any] = None,
                     calc_perplexity: bool = True) -> List[Dict]:
        """Process a batch of sentences with optional perplexity calculation"""
        # Process POS tags for all sentences using nlp.pipe
        pos_tags_results = []
        for doc in self.nlp.pipe(sentences):
            pos_tags = [(token.text, token.pos_) for token in doc]
            pos_tags_results.append(pos_tags)

        # Process perplexity separately if needed
        if calc_perplexity and model is not None and tokenizer is not None:
            perplexities = self.calculate_perplexity_batch(sentences, model, tokenizer)
        else:
            perplexities = [float('nan')] * len(sentences)

        # Combine results
        results = []
        for i, pos_tags in enumerate(pos_tags_results):
            result = {
                'pos_tags': pos_tags,
                'perplexity': perplexities[i]
            }
            results.append(result)

        return results

    def process_dataframe(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enhanced processing with minimal DataFrame copying"""
        start_time = time.time()
        
        # Create view of sentence column without copying
        sentences = df['sentence'].values
        
        # Remove long sentences if we're calculating perplexity
        valid_indices = None
        if kwargs.get('calc_perplexity') and kwargs.get('tokenizer') is not None:
            # Get indices of sentences that aren't too long
            token_lengths = np.array([len(kwargs['tokenizer'].encode(s)) for s in sentences])
            valid_indices = np.where(token_lengths <= 200)[0]
            sentences = sentences[valid_indices]
        
        # Initialize result columns
        results = {'pos_tags': [], 'perplexity': []}
        
        # Process in batches with a single progress bar
        total_sentences = len(sentences)
        with tqdm(total=total_sentences, desc="Processing sentences") as pbar:
            for i in range(0, total_sentences, self.max_batch_size):
                end_idx = min(i + self.max_batch_size, total_sentences)
                # Process batch
                chunk_sentences = sentences[i:end_idx]
                chunk_results = self.process_batch(chunk_sentences, **kwargs)
                
                # Collect results
                for res in chunk_results:
                    results['pos_tags'].append(res['pos_tags'])
                    results['perplexity'].append(res['perplexity'])
                
                pbar.update(len(chunk_sentences))
        
        # Update original DataFrame directly
        if valid_indices is not None:
            # Create temporary Series for the results
            pos_tags_series = pd.Series(None, index=df.index)
            perplexity_series = pd.Series(float('nan'), index=df.index)
            
            # Update only the valid indices
            pos_tags_series.iloc[valid_indices] = results['pos_tags']
            perplexity_series.iloc[valid_indices] = results['perplexity']
            
            # Assign to DataFrame
            df['pos_tags'] = pos_tags_series
            df['perplexity'] = perplexity_series
        else:
            # Assign directly if we processed all sentences
            df['pos_tags'] = results['pos_tags']
            df['perplexity'] = results['perplexity']
        
        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(f"Total processing time: {processing_time:.2f} seconds")
        logging.info(f"Average time per sentence: {processing_time/total_sentences:.3f} seconds")
        
        return df

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            dataset_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            return pd.read_csv(dataset_path)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise 

    def save_to_csv(self, df, output_path):
        """
        Save the processed DataFrame to a CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Path where the CSV file will be saved
        """
        try:
            # Get original filename without extension and directory
            original_filename = os.path.splitext(os.path.basename(output_path))[0]
            
            # Extract the base directory and model info
            output_dir = os.path.dirname(output_path)
            model_info = f"_{df['perplexity'].notna().sum()}_perplexity_scores"
            
            # Create new filename with original name and model info
            new_filename = f"{original_filename}{model_info}.csv"
            new_path = os.path.join(output_dir, new_filename)
            
            df.to_csv(new_path, index=False)
            print(f"Successfully saved results to {new_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

    def remove_long_sentences(self, df: pd.DataFrame, tokenizer: Any, max_tokens: int = 200) -> pd.DataFrame:
        """Remove sentences that exceed the specified token length using vectorized operations."""
        # Calculate token lengths without copying
        token_lengths = df['sentence'].apply(lambda x: len(tokenizer.encode(x)))
        
        # Filter sentences without explicit copy
        return df[token_lengths <= max_tokens]
