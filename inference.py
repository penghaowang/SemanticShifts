import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Union, Any, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import json
import os
from datetime import datetime
import traceback
from datasets import Dataset

# Import custom logger configuration
from logger_config import setup_logger, TRACE

class ModelInference:
    def __init__(
        self,
        model_name: str,
        layer_indices: Optional[List[int]] = None,
        batch_size: int = 16,
        num_beams: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 100,
        prompt_template: List[Dict[str, str]] = None,
        extraction_method: str = "input_last_token",  # Can be a single method or "all"
        use_all_layers: bool = True,
        target_word: Optional[str] = None,
        target_pos: Optional[str] = None,
        saved_dataset_dir: Optional[str] = None,
        output_dir: str = "outputs",
        save_hidden_states: bool = True,
        log_file: str = "logs/inference.log",
        use_8bit: bool = False
    ):
        """
        Initialize the model inference class
        
        Args:
            model_name: Model name or path
            layer_indices: List of layer indices to extract, None means all layers
            batch_size: Batch size
            num_beams: Number of beams for beam search
            temperature: Generation temperature
            max_new_tokens: Maximum number of generated tokens
            prompt_template: List of prompt templates
            extraction_method: Hidden states extraction method
                - input_last_token: Extract the last token of the input
                - eos_token: Add EOS token after input and extract
                - input_mean: Perform mean pooling on all tokens of the input sentence
                - output_mean: Perform mean pooling on the generated content
                - output_eos: Extract the EOS token of the generated content
                - all: Extract hidden states using all methods (optimized calculation)
            use_all_layers: Whether to use hidden states from all layers
            target_word: Target word for extraction
            target_pos: Target position for extraction
            saved_dataset_dir: Path to the saved dataset directory
            output_dir: Output directory path
            save_hidden_states: Whether to save hidden states
            log_file: Log file path
            use_8bit: Whether to use 8-bit quantization
        """
        # Set up module-specific logger
        self.logger = setup_logger('model_inference', log_file)
        
        self.model_name = model_name
        self.layer_indices = layer_indices
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt_template = prompt_template
        self.extraction_method = extraction_method
        self.use_all_layers = use_all_layers
        self.target_word = target_word
        self.target_pos = target_pos
        self.saved_dataset_dir = saved_dataset_dir
        self.output_dir = output_dir
        self.save_hidden_states = save_hidden_states
        self.use_8bit = use_8bit
        
        # Initialize model and tokenizer
        self.logger.info(f"Loading model: {model_name}, Using 8-bit quantization: {use_8bit}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            load_in_8bit=use_8bit
        )
        
        # Set pad_token_id and eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.debug("Set pad_token_id to eos_token_id")
            
        # Log model configuration
        self.logger.info("=== Model Configuration ===")
        self.logger.info(f"  Model Name: {model_name}")
        self.logger.info(f"  Layer Indices: {layer_indices}")
        self.logger.info(f"  Batch Size: {batch_size}")
        self.logger.info(f"  Number of Beams: {num_beams}")
        self.logger.info(f"  Generation Temperature: {temperature}")
        self.logger.info(f"  Max Generated Tokens: {max_new_tokens}")
        self.logger.info(f"  Extraction Method: {extraction_method}")
        self.logger.info(f"  Use All Layers: {use_all_layers}")
        self.logger.info(f"  Target Word: {target_word}")
        self.logger.info(f"  Target Position: {target_pos}")
        if saved_dataset_dir:
            self.logger.info(f"  Saved Dataset Directory: {saved_dataset_dir}")
        
        # Added logic for layer count validation
        self.total_layers = self.model.config.num_hidden_layers
        if layer_indices and any(idx >= self.total_layers for idx in self.layer_indices):
            error_msg = f"Layer index exceeds total number of layers ({self.total_layers})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Model initialization complete, total layers: {self.total_layers}")
        
    def format_prompt(self, sentence: str, word: str) -> List[Dict[str, str]]:
        """
        Format prompt according to the template
        
        Args:
            sentence: Input sentence
            word: Target word
            
        Returns:
            List of formatted prompts
        """
        formatted_prompt = []
        for message in self.prompt_template:
            formatted_message = {
                'role': message['role'],
                'content': message['content'].format(
                    sentence=sentence,
                    word=word
                )
            }
            formatted_prompt.append(formatted_message)
        return formatted_prompt
        
    def load_saved_dataset(self) -> Optional[Dataset]:
        """Load dataset from the saved directory"""
        if not self.saved_dataset_dir or not self.target_word or not self.target_pos:
            return None
            
        dataset_path = os.path.join(self.saved_dataset_dir, f"{self.target_word}_{self.target_pos}")
        if os.path.exists(dataset_path):
            try:
                dataset = Dataset.load_from_disk(dataset_path)
                self.logger.info(f"Loaded dataset from {dataset_path}")
                self.logger.info(f"Dataset size: {len(dataset)} examples")
                return dataset
            except Exception as e:
                self.logger.error(f"Error loading dataset: {e}")
                return None
        else:
            self.logger.warning(f"No saved dataset found at {dataset_path}")
            return None
            
    def run_inference(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run inference on the input data
        
        Args:
            df: DataFrame containing the 'sentence' column, if None, try loading from saved dataset
            
        Returns:
            DataFrame containing inference results
        """
        # If DataFrame is not provided, try loading the saved dataset
        if df is None:
            dataset = self.load_saved_dataset()
            if dataset is None:
                raise ValueError("No input data provided and failed to load saved dataset")
            df = pd.DataFrame(dataset)
        
        results = []
        all_hidden_states = {}  # Dictionary to collect hidden states by method
        
        # Process data in batches
        for i in tqdm(range(0, len(df), self.batch_size)):
            batch_df = df.iloc[i:i + self.batch_size]
            
            # Prepare batch data
            batch_prompts = []
            for _, row in batch_df.iterrows():
                prompt = self.format_prompt(
                    row['sentence'], 
                    self.target_word if self.target_word else row.get('word', '')
                )
                batch_prompts.append(prompt)
            
            # Get hidden states and generation results
            batch_results = self._process_batch(batch_prompts, batch_df)
            
            # Extract and collect hidden states
            for result in batch_results:
                for key, value in list(result.items()):
                    if isinstance(value, torch.Tensor):
                        # Log tensor shape for debugging
                        #self.logger.info(f"Collecting tensor {key}, shape: {value.shape}")
                        
                        if key not in all_hidden_states:
                            all_hidden_states[key] = []
                        
                        # Ensure tensor is on CPU and is float type
                        try:
                            tensor_to_save = value.detach().cpu().float()
                            
                            # Check if tensor is valid
                            if torch.isnan(tensor_to_save).any() or torch.isinf(tensor_to_save).any():
                                #self.logger.warning(f"Tensor {key} contains NaN or Inf values, will be replaced with zero tensor")
                                tensor_to_save = torch.zeros_like(tensor_to_save)
                                
                            all_hidden_states[key].append(tensor_to_save)
                            
                            # Remove tensor from results to avoid memory issues
                            result[key] = f"tensor_collected_for_{key}"
                        except Exception as e:
                            self.logger.error(f"Error processing tensor {key}: {str(e)}")
                            self.logger.error(traceback.format_exc())
            
            results.extend(batch_results)
        
        results_df = pd.DataFrame(results)
        
        # Save combined hidden states
        if self.save_hidden_states:
            self.logger.info("Saving combined hidden states...")
            self.logger.info(f"Collected hidden state methods: {list(all_hidden_states.keys())}")
            
            # Check if any hidden states were collected
            if not all_hidden_states:
                self.logger.warning("No hidden states collected, cannot save")
                return results_df
                
            # Check the number of tensors collected for each method
            for method, tensors in all_hidden_states.items():
                self.logger.info(f"Method {method} collected {len(tensors)} tensors")
                if not tensors:
                    self.logger.warning(f"Method {method} did not collect any tensors")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"hidden_states_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            saved_files = []  # Record successfully saved files
            
            for method, tensors in all_hidden_states.items():
                self.logger.info(f"Combining hidden states for method '{method}'...")
                if not tensors:
                    self.logger.warning(f"No tensors to combine for method '{method}'")
                    continue
                    
                combined_tensor = torch.cat(tensors, dim=0) # Combine tensors along the batch dimension
                safe_target_word = str(self.target_word).replace(':', '_') if self.target_word else "unknown"
                safe_target_pos = str(self.target_pos).replace(':', '_') if self.target_pos else "unknown"
                file_path = os.path.join(output_dir, f"hidden_states_{safe_target_word}_{safe_target_pos}_{method}.pt")
                
                # Check if tensor is valid before saving
                if torch.isnan(combined_tensor).any():
                    #self.logger.warning(f"Tensor for method {method} contains NaN values, skipping save")
                    continue
                    
                if torch.isinf(combined_tensor).any():
                    #self.logger.warning(f"Tensor for method {method} contains Inf values, skipping save")
                    continue
                
                torch.save(combined_tensor, file_path)
                saved_files.append(file_path)
                self.logger.info(f"Saved {len(tensors)} hidden states to: {file_path}")
            
            if saved_files:
                self.logger.info(f"Successfully saved {len(saved_files)} hidden states files")
                self.logger.info(f"Hidden states saved to directory: {output_dir}")
            else:
                self.logger.warning("No hidden states files saved")
        
        return results_df
    
    def _process_batch(self, batch_prompts, batch_df):
        batch_results = []
        
        for prompt, (_, row) in zip(batch_prompts, batch_df.iterrows()):
            # Convert prompt list to string
            if isinstance(prompt, list):
                # If it's a message list format (like ChatML), convert to string
                prompt_str = ""
                for message in prompt:
                    role = message.get('role', '')
                    content = message.get('content', '')
                    prompt_str += f"{role}: {content}\n"
            else:
                # If it's already a string, use it directly
                prompt_str = prompt
            
            # Process input
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
            input_length = inputs['input_ids'].shape[1]  # Record original input length
            
            # Decide whether generation is needed based on extraction method
            need_generation = self.extraction_method in ["output_mean", "output_eos", "all"]
            
            try:
                if need_generation:
                    # Perform generation and save generation info
                    with torch.no_grad():
                        generation_outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            num_beams=self.num_beams,
                            temperature=self.temperature,
                            output_hidden_states=True,
                            return_dict_in_generate=True
                        )
                    
                    # Save the full generated sequence and original input length
                    self.generation_outputs = generation_outputs
                    self.input_length = input_length
                    
                    # Extract hidden states from the generated part
                    output_hidden_states = self._extract_hidden_states(
                        generation_outputs.hidden_states,
                        generation_outputs.sequences,
                        method=self.extraction_method
                    )
                    
                    # Generated text
                    generated_text = self.tokenizer.decode(
                        generation_outputs.sequences[0][input_length:], 
                        skip_special_tokens=True
                    )
                    
                    # Check if regeneration is needed (if extraction method returns need_generation flag)
                    if isinstance(output_hidden_states, dict) and output_hidden_states.get("need_generation", False):
                        #self.logger.info("Extraction method requires generation process, but an issue was detected, attempting regeneration")
                        # Already generated, directly re-extract
                        output_hidden_states = self._extract_hidden_states(
                            generation_outputs.hidden_states,
                            generation_outputs.sequences,
                            method=self.extraction_method
                        )
                else:
                    # If generation is not needed, directly extract input hidden states
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True)
                    
                    input_hidden_states = self._extract_hidden_states(
                        outputs.hidden_states,
                        inputs['input_ids'],
                        method=self.extraction_method
                    )
                    
                    generated_text = "[No generation needed]"
                    output_hidden_states = input_hidden_states
                
                # Build result
                result = {
                    'sentence': row['sentence'],
                    'generated_text': generated_text,
                }
                
                # Add hidden states to the result
                if output_hidden_states and isinstance(output_hidden_states, dict) and 'hidden_states' in output_hidden_states:
                    result[f'hidden_states_{self.extraction_method}'] = output_hidden_states['hidden_states']
                    
                    # Add metadata
                    if 'metadata' in output_hidden_states:
                        result[f'metadata_{self.extraction_method}'] = output_hidden_states['metadata']
                    
                    # Add position information
                    if 'positions' in output_hidden_states:
                        result[f'positions_{self.extraction_method}'] = output_hidden_states['positions']
                
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")
                self.logger.error(traceback.format_exc())
                # Add error info to result
                batch_results.append({
                    'sentence': row['sentence'],
                    'generated_text': f"[Error: {str(e)}]",
                    'error': str(e)
                })
        
        return batch_results
    
    def _process_input_last_token(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process hidden state extraction for the input_last_token method
        
        Args:
            hidden_states: Model's hidden states
            input_ids: Input token IDs
            layer_indices: List of layer indices to extract
            
        Returns:
            Dictionary containing processed hidden states and metadata, or None (on failure)
        """
        try:
            # Validate input
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states is empty")
                return None
                
            if input_ids.numel() == 0:
                #self.logger.error("input_ids is empty")
                return None
                
            # Handle case where layer_indices is None
            if layer_indices is None:
                if self.use_all_layers:
                    layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # Default to using the first layer
                
            states = []
            valid_layers = []
            # Initialize metadata only for extraction_positions
            metadata = {}

            # Get the position of the last non-EOS token
            last_token_pos = (input_ids != self.tokenizer.pad_token_id).sum(dim=1) - 1
            
            # Ensure position index is valid
            last_token_pos = torch.clamp(last_token_pos, min=0, max=input_ids.size(1)-1)
            
            # If there is an EOS token, take the position before it
            eos_mask = (input_ids[torch.arange(input_ids.size(0)), last_token_pos] == self.tokenizer.eos_token_id)
            if eos_mask.any():
                #self.logger.info(f"Found {eos_mask.sum().item()} sequences where the last token is EOS, will take the preceding position")
                last_token_pos[eos_mask] = torch.clamp(last_token_pos[eos_mask] - 1, min=0)

            # Record extraction position information
            metadata['extraction_positions'] = last_token_pos.tolist()
            #self.logger.info(f"Input sequence length: {input_ids.shape[1]}")
            #self.logger.info(f"Actual extraction positions: {last_token_pos.tolist()}")
            #self.logger.info(f"Number of hidden state layers: {len(hidden_states)}")

            # Determine layers to process
            valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
            if not valid_layer_indices:
                #self.logger.error(f"No valid layer indices. Requested indices: {layer_indices}, Available layers: {len(hidden_states)}")
                return None
                
            #self.logger.info(f"Will process the following layers: {valid_layer_indices}")

            for layer_idx in valid_layer_indices:
                try:
                    # Get hidden states for the current layer
                    layer_states = hidden_states[layer_idx]
                    
                    # Handle different types of hidden states
                    if isinstance(layer_states, tuple):
                        #self.logger.info(f"Hidden states for layer {layer_idx} are tuple type, attempting conversion")
                        layer_states = torch.stack(list(layer_states))
                    elif not isinstance(layer_states, torch.Tensor):
                        #self.logger.error(f"Unsupported hidden state type for layer {layer_idx}: {type(layer_states)}")
                        continue
                    
                    # Check shape
                    #self.logger.info(f"Shape of hidden states for layer {layer_idx}: {layer_states.shape}")
                    
                    if layer_states.dim() < 3:
                        #self.logger.error(f"Insufficient dimensions for hidden states in layer {layer_idx}: {layer_states.dim()}")
                        continue
                        
                    if layer_states.size(0) != input_ids.size(0):
                        #self.logger.error(f"Batch size mismatch for layer {layer_idx}: {layer_states.size(0)} vs {input_ids.size(0)}")
                        continue

                    # Extract hidden states for the last token in each sequence
                    batch_states = []
                    for batch_idx, pos_idx in enumerate(last_token_pos):
                        # Prevent position index from going out of bounds
                        safe_pos = min(pos_idx.item(), layer_states.size(1)-1)
                        token_state = layer_states[batch_idx, safe_pos]
                        
                        # Check the extracted tensor
                        if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                            #self.logger.warning(f"Tensor at batch {batch_idx}, position {safe_pos} contains NaN or Inf values, will be replaced with zero")
                            token_state = torch.zeros_like(token_state)
                            
                        batch_states.append(token_state)
                    
                    # Stack results along the batch dimension
                    batch_states = torch.stack(batch_states)
                    states.append(batch_states)
                    valid_layers.append(layer_idx)

                    # Record statistics
                    # metadata['last_token_stats'][f'layer_{layer_idx}'] = {
                    #     'mean': float(batch_states.mean().item()),
                    #     'std': float(batch_states.std().item()),
                    #     'min': float(batch_states.min().item()),
                    #     'max': float(batch_states.max().item())
                    # }

                except Exception as e:
                    #self.logger.error(f"Error processing layer {layer_idx}: {str(e)}")
                    #self.logger.error(traceback.format_exc())
                    # metadata['processing_info'].append(f"Layer {layer_idx} processing failed: {str(e)}")
                    continue

            if not states:
                #self.logger.error("No valid states collected")
                return None

            try:
                # Stack results from all layers
                stacked_states = torch.stack(states)  # [num_layers, batch_size, hidden_size]
                
                # Build result
                result = {
                    'hidden_states': stacked_states,
                    'valid_layers': valid_layers,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids],
                    'positions': last_token_pos.cpu().numpy()
                }

                #self.logger.info(f"Final hidden_states shape: {result['hidden_states'].shape}")
                #self.logger.info(f"Processed {len(valid_layers)}/{len(valid_layer_indices)} layers")

                return result
                
            except Exception as e:
                #self.logger.error(f"Error stacking layer states: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"Error processing input_last_token: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _extract_hidden_states(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        method: str = "input_last_token"
    ) -> Optional[Dict[str, Any]]:
        """Enhanced version for extracting hidden states"""
        # Check structure of hidden_states
        #self.logger.info(f"Extraction method: {method}")
        #self.logger.info(f"Hidden states type: {type(hidden_states)}")
        #self.logger.info(f"Hidden states length: {len(hidden_states)}")
        
        # Check the first element to further understand the structure
        if len(hidden_states) > 0:
            #self.logger.info(f"First element type: {type(hidden_states[0])}, shape: {hidden_states[0].shape if isinstance(hidden_states[0], torch.Tensor) else None}")
            pass
        
        # Improved logic for detecting generation model hidden states
        is_generation_output = False
        input_length = getattr(self, 'input_length', None)
        
        # Check if it's the output structure of generation model
        if isinstance(hidden_states, list) and len(hidden_states) > 0:
            first_element = hidden_states[0]
            if isinstance(first_element, (tuple, list)) and len(first_element) > 0:
                is_generation_output = True
                #self.logger.info("Detected generation model hidden state structure (list contains tuples or lists)")
        elif isinstance(hidden_states, tuple) and method in ["output_mean", "output_eos"]:
            # If it's a tuple, and method is output_mean or output_eos, special handling might be needed
            #self.logger.warning("hidden_states is a tuple, might be generation model hidden states, attempting to process")
            # Check if the elements in the tuple are tensors and the number matches the number of layers
            if len(hidden_states) == self.model.config.num_hidden_layers:
                # Assume this is a normal hidden state tuple, not a list of generation steps
                #self.logger.info("hidden_states is a hidden state tuple for layers, not a list of generation steps")
                is_generation_output = False
            else:
                # Might be a tuple of generation steps, attempt to convert to list
                #self.logger.info("Attempting to convert tuple to list of generation steps")
                hidden_states = list(hidden_states)
                is_generation_output = True
        
        # Check if generation sequence and input length info exists
        has_generation_info = hasattr(self, 'generation_outputs') and hasattr(self, 'input_length')
        
        if method in ["output_mean", "output_eos"] and not is_generation_output and not has_generation_info:
            #self.logger.warning(f"Method {method} requires generation model hidden states, but the correct structure was not detected")
            pass
        
        # Call the corresponding processing function based on the extraction method
        try:
            result = None
            if method == "input_last_token":
                result = self._process_input_last_token(hidden_states, input_ids)
            elif method == "eos_token":
                result = self._process_eos_token(hidden_states, input_ids)
            elif method == "input_mean":
                result = self._process_input_mean(hidden_states, input_ids)
            elif method == "output_mean":
                result = self._process_output_mean(hidden_states, input_ids)
            elif method == "output_eos":
                result = self._process_output_eos(hidden_states, input_ids)
            else:
                #self.logger.error(f"Unsupported extraction method: {method}")
                return None
            
            if result is None:
                #self.logger.error(f"Method {method} returned None")
                return None
            
            return result
        except Exception as e:
            #self.logger.error(f"Error extracting hidden states: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_eos_token(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Process hidden state extraction for the eos_token method
        
        Args:
            hidden_states: Model's hidden states
            input_ids: Input token IDs
            layer_indices: List of layer indices to extract
            
        Returns:
            Dictionary containing processed hidden states and metadata, or None (on failure)
        """
        try:
            # Validate input
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states is empty")
                return None
                
            if input_ids.numel() == 0:
                #self.logger.error("input_ids is empty")
                return None
            
            # Handle case where layer_indices is None
            if layer_indices is None:
                if self.use_all_layers:
                    layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # Default to using the first layer
                
            states = []
            valid_layers = []
            # REMOVED: metadata initialization with stats
            # metadata = {
            #     'eos_stats': {},
            #     'processing_info': []
            # }
            
            # Determine layers to process
            valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
            if not valid_layer_indices:
                #self.logger.error(f"No valid layer indices. Requested indices: {layer_indices}, Available layers: {len(hidden_states)}")
                return None
                
            #self.logger.info(f"Will process the following layers: {valid_layer_indices}")
            #self.logger.info(f"Input sequence shape: {input_ids.shape}")

            # Find positions of EOS token
            eos_positions = (input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)
            if eos_positions[0].numel() > 0:
                #self.logger.info(f"Found EOS token positions: Batch indices={eos_positions[0].tolist()}, Sequence positions={eos_positions[1].tolist()}")
                # Process each sequence separately
                eos_pos_by_batch = {}
                for i in range(len(eos_positions[0])):
                    batch_idx = eos_positions[0][i].item()
                    pos = eos_positions[1][i].item()
                    if batch_idx not in eos_pos_by_batch:
                        eos_pos_by_batch[batch_idx] = pos
            else:
                #self.logger.warning("No EOS tokens found, will use end of sequence position")
                eos_pos_by_batch = {}

            for layer_idx in valid_layer_indices:
                try:
                    layer_states = hidden_states[layer_idx]
                    
                    # Handle different types of hidden states
                    if isinstance(layer_states, tuple):
                        #self.logger.info(f"Hidden states for layer {layer_idx} are tuple type, attempting conversion")
                        layer_states = torch.stack(list(layer_states))
                    elif not isinstance(layer_states, torch.Tensor):
                        #self.logger.error(f"Unsupported hidden state type for layer {layer_idx}: {type(layer_states)}")
                        continue
                        
                    # Check shape
                    #self.logger.info(f"Shape of hidden states for layer {layer_idx}: {layer_states.shape}")
                    
                    if layer_states.dim() < 3:
                        #self.logger.error(f"Insufficient dimensions for hidden states in layer {layer_idx}: {layer_states.dim()}")
                        continue
                        
                    if layer_states.size(0) != input_ids.size(0):
                        #self.logger.error(f"Batch size mismatch for layer {layer_idx}: {layer_states.size(0)} vs {input_ids.size(0)}")
                        continue
                    
                    # Extract hidden states for the EOS token in each sequence
                    batch_states = []
                    for batch_idx in range(input_ids.size(0)):
                        if batch_idx in eos_pos_by_batch:
                            # use found EOS position
                            pos = eos_pos_by_batch[batch_idx]
                        else:
                            # If no EOS token, use the last non-padding token
                            pos = (input_ids[batch_idx] != self.tokenizer.pad_token_id).sum().item() - 1
                            pos = max(0, pos)  # Ensure position is not negative
                        
                        # Ensure position is within valid range
                        safe_pos = min(pos, layer_states.size(1) - 1)
                        token_state = layer_states[batch_idx, safe_pos]
                        
                        # Check the extracted tensor
                        if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                            #self.logger.warning(f"Tensor at layer {layer_idx}, batch {batch_idx}, position {safe_pos} contains NaN or Inf, will replace with zero")
                            token_state = torch.zeros_like(token_state)
                            
                        batch_states.append(token_state)
                    
                    # Stack results along the batch dimension
                    batch_states = torch.stack(batch_states)
                    states.append(batch_states)
                    valid_layers.append(layer_idx)
                    
                    # Record statistics
                    # metadata['eos_stats'][f'layer_{layer_idx}'] = {
                    #     'mean': float(batch_states.mean().item()),
                    #     'std': float(batch_states.std().item()),
                    #     'min': float(batch_states.min().item()),
                    #     'max': float(batch_states.max().item())
                    # }

                except Exception as e:
                    #self.logger.error(f"Error processing layer {layer_idx}: {str(e)}")
                    #self.logger.error(traceback.format_exc())
                    # REMOVED: processing_info append
                    # metadata['processing_info'].append(f"Layer {layer_idx} processing failed: {str(e)}")
                    continue

            if not states:
                #self.logger.error("No valid states collected")
                return None
            
            # Build result
            result = {
                'hidden_states': torch.stack(states),  # [num_layers, batch_size, hidden_size]
                'valid_layers': valid_layers,
                'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
            }
            
            #self.logger.info(f"Final hidden_states shape: {result['hidden_states'].shape}")
            #self.logger.info(f"Processed {len(valid_layers)}/{len(valid_layer_indices)} layers")
            
            return result
        except Exception as e:
            #self.logger.error(f"Error processing eos_token: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_input_mean(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Process hidden state extraction for the input_mean method"""
        try:
            # Validate input
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states is empty")
                return None
            
            if input_ids.numel() == 0:
                #self.logger.error("input_ids is empty")
                return None
            
            # Handle case where layer_indices is None
            if layer_indices is None:
                if self.use_all_layers:
                    layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # Default to using the first layer
            
            states = []
            all_mean_states = []  # Add this variable
            valid_layers = []
            
            # Create attention mask to ignore padding tokens
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
            #self.logger.info(f"Input sequence shape: {input_ids.shape}")
            #self.logger.info(f"Attention mask shape: {attention_mask.shape}")
            
            # Determine layers to process
            valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
            if not valid_layer_indices:
                #self.logger.error(f"No valid layer indices. Requested indices: {layer_indices}, Available layers: {len(hidden_states)}")
                return None
            
            #self.logger.info(f"Will process the following layers: {valid_layer_indices}")

            for layer_idx in valid_layer_indices:
                try:
                    layer_states = hidden_states[layer_idx]
                    
                    # Handle different types of hidden states
                    if isinstance(layer_states, tuple):
                        #self.logger.info(f"Hidden states for layer {layer_idx} are tuple type, attempting conversion")
                        layer_states = torch.stack(list(layer_states))
                    elif not isinstance(layer_states, torch.Tensor):
                        #self.logger.error(f"Unsupported hidden state type for layer {layer_idx}: {type(layer_states)}")
                        continue
                    
                    # Check shape
                    #self.logger.info(f"Shape of hidden states for layer {layer_idx}: {layer_states.shape}")
                    
                    if layer_states.dim() < 3:
                        #self.logger.error(f"Insufficient dimensions for hidden states in layer {layer_idx}: {layer_states.dim()}")
                        continue
                        
                    if layer_states.size(0) != input_ids.size(0):
                        #self.logger.error(f"Batch size mismatch for layer {layer_idx}: {layer_states.size(0)} vs {input_ids.size(0)}")
                        continue
                    
                    # Calculate mean (ignoring padding tokens)
                    try:
                        # Expand attention mask to match hidden state dimensions
                        expanded_mask = attention_mask.unsqueeze(-1)
                        
                        # Apply mask
                        masked_states = layer_states * expanded_mask
                        
                        # Calculate mean for each sequence
                        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                        mean_states = masked_states.sum(dim=1) / sum_mask
                        
                        # Check for NaN or Inf
                        if torch.isnan(mean_states).any() or torch.isinf(mean_states).any():
                            self.logger.warning(f"Mean hidden states for layer {layer_idx} contain NaN or Inf, will replace with zero")
                            mean_states = torch.zeros_like(mean_states)
                        
                        all_mean_states.append(mean_states)
                        valid_layers.append(layer_idx)
                        
                        # Modify log record for shape of mean hidden states
                        #self.logger.info(f"Shape of mean hidden states for layer {layer_idx}: {mean_states.shape}")
                    except Exception as e:
                        #self.logger.error(f"Error calculating mean state for layer {layer_idx}: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        continue
                    
                    states.append(mean_states)
                    
                    # Record statistics
                    # metadata['mean_stats'][f'layer_{layer_idx}'] = {
                    #     'mean': float(mean_states.mean().item()),
                    #     'std': float(mean_states.std().item()),
                    #     'min': float(mean_states.min().item()),
                    #     'max': float(mean_states.max().item())
                    # }

                except Exception as e:
                    #self.logger.error(f"Error processing layer {layer_idx}: {str(e)}")
                    #self.logger.error(traceback.format_exc())
                    # REMOVED: processing_info append
                    # metadata['processing_info'].append(f"Layer {layer_idx} processing failed: {str(e)}")
                    continue

            if not states:
                #self.logger.error("No valid states collected")
                return None
            
            # Stack all layers
            try:
                stacked_all_layers = torch.stack(all_mean_states)
                
                # Build result
                result = {
                    'hidden_states': stacked_all_layers,  # [num_layers, batch_size, hidden_size]
                    'valid_layers': valid_layers,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
                }
                
                #self.logger.info(f"Final mean hidden_states shape: {result['hidden_states'].shape}")
                #self.logger.info(f"Processed {len(valid_layers)}/{len(valid_layer_indices)} layers")
                
                return result
            except Exception as e:
                #self.logger.error(f"Error building result: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"Error processing input_mean: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_output_mean(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None,
        is_generation_output: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Process hidden state extraction for the output_mean method"""
        try:
            # Validate input
            if not hidden_states:
                #self.logger.error("hidden_states is empty")
                return None
                
            if input_ids.numel() == 0:
                #self.logger.error("input_ids is empty")
                return None
        
            # Get input sequence length
            input_length = getattr(self, 'input_length', None)
            if input_length is None or input_length <= 0:
                #self.logger.warning("Cannot determine valid input length, using heuristic")
                input_length = input_ids.shape[1] // 2 # Assume generated part is half the total length
        
            # Get total sequence length
            total_length = input_ids.shape[1]  # Directly use length of input_ids as total length
        
            #self.logger.info(f"Input length: {input_length}, Total length: {total_length}, Generated part length: {total_length - input_length}")
        
            # Check if there is a generated part
            if total_length <= input_length:
                #self.logger.error(f"Total length ({total_length}) is not greater than input length ({input_length}), cannot extract generated part")
                return None
        
            # Check structure of hidden_states
            #self.logger.info(f"hidden_states type: {type(hidden_states)}")
            #self.logger.info(f"hidden_states length: {len(hidden_states)}")
            #self.logger.info(f"Is generation model output: {is_generation_output}")
        
            all_mean_states = []
            valid_layers = []
            
            # Handle case where layer_indices is None
            if layer_indices is None:
                if self.use_all_layers:
                    # If using all layers, determine layer count based on hidden state structure
                    if is_generation_output and len(hidden_states) > 0 and isinstance(hidden_states[0], (tuple, list)):
                        layer_indices = list(range(len(hidden_states[0])))
                    else:
                        layer_indices = list(range(len(hidden_states)))
                else:
                    # If not using all layers but no layer indices specified, use default value
                    layer_indices = [0]  # Default to using the first layer
                    
            #self.logger.info(f"Using layer indices: {layer_indices}")
            
            if is_generation_output:
                # Process generation model hidden state structure
                #self.logger.info("Processing generation model hidden state structure")
                
                # Determine valid layer indices
                if len(hidden_states) > 0 and isinstance(hidden_states[0], (tuple, list)):
                    num_layers = len(hidden_states[0])
                    valid_layer_indices = [idx for idx in layer_indices if idx < num_layers]
                else:
                    #self.logger.error("Generation model hidden state structure does not match expectations")
                    return None
                
                if not valid_layer_indices:
                    #self.logger.error(f"No valid network layer indices. Requested indices: {layer_indices}, Available layers: {num_layers}")
                    return None
                
                #self.logger.info(f"Will process the following network layers: {valid_layer_indices}")
                
                # Process all generation steps for each layer
                for layer_idx in valid_layer_indices:
                    try:
                        # Collect hidden states for all generation steps in this layer
                        layer_states = []
                        
                        for step_idx, step_hidden_states in enumerate(hidden_states):
                            if layer_idx < len(step_hidden_states):
                                current_layer_state = step_hidden_states[layer_idx]
                                
                                # Handle different types of hidden states
                                if isinstance(current_layer_state, tuple):
                                    #self.logger.info(f"Hidden states for step {step_idx}, network layer {layer_idx} are tuple type, attempting to get first element")
                                    if len(current_layer_state) > 0:
                                        current_layer_state = current_layer_state[0]
                                    else:
                                        continue
                                
                                if not isinstance(current_layer_state, torch.Tensor):
                                    #self.logger.warning(f"Unsupported hidden state type for step {step_idx}, network layer {layer_idx}: {type(current_layer_state)}")
                                    continue
                                
                                # Check shape
                                if len(current_layer_state.shape) != 3:
                                    #self.logger.warning(f"Incorrect dimension for hidden states at step {step_idx}, network layer {layer_idx}: {len(current_layer_state.shape)}, expected 3")
                                    continue
                                
                                # For generation models, each step has only one token, we directly collect hidden states for all steps
                                layer_states.append(current_layer_state)
                        
                        if not layer_states:
                            #self.logger.warning(f"No valid hidden states collected for network layer {layer_idx}")
                            continue
                        
                        # Concatenate hidden states from all steps
                        concatenated_states = torch.cat(layer_states, dim=1)
                        #self.logger.info(f"Shape of concatenated hidden states for network layer {layer_idx}: {concatenated_states.shape}")
                        
                        # Extract generated part (if input length info is available)
                        if input_length > 0 and concatenated_states.shape[1] > input_length:
                            generated_part = concatenated_states[:, input_length:, :]
                            #self.logger.info(f"Extracting generated part for network layer {layer_idx}, shape: {generated_part.shape}")
                        else:
                            # If no explicit input length, assume all steps are generated
                            generated_part = concatenated_states
                            #self.logger.info(f"Using all hidden states for network layer {layer_idx}, shape: {generated_part.shape}")
                        
                        # Calculate mean
                        mean_output_hidden_states = torch.mean(generated_part, dim=1)
                        
                        # Check for NaN or Inf
                        if torch.isnan(mean_output_hidden_states).any() or torch.isinf(mean_output_hidden_states).any():
                            #self.logger.warning(f"Mean hidden states for network layer {layer_idx} contain NaN or Inf, will replace with zero")
                            mean_output_hidden_states = torch.zeros_like(mean_output_hidden_states)
                        
                        all_mean_states.append(mean_output_hidden_states)
                        valid_layers.append(layer_idx)
                    except Exception as e:
                        #self.logger.error(f"Error processing generation model network layer {layer_idx}: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        continue
            else:
                # Process standard model hidden states
                #self.logger.info("Processing standard model hidden states")
            
                valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
                if not valid_layer_indices:
                    #self.logger.error(f"No valid network layer indices. Requested indices: {layer_indices}, Available layers: {len(hidden_states)}")
                    return None
                
                #self.logger.info(f"Will process the following network layers: {valid_layer_indices}")
            
                # Process each layer
                for layer_idx in valid_layer_indices:
                    try:
                        layer_states = hidden_states[layer_idx]
                    
                        # Handle different types of hidden states
                        if isinstance(layer_states, tuple):
                            #self.logger.info(f"Hidden states for network layer {layer_idx} are tuple type, attempting to get first element")
                            if len(layer_states) > 0:
                                layer_states = layer_states[0]
                            else:
                                #self.logger.warning(f"Hidden states for network layer {layer_idx} is an empty tuple")
                                continue
                        
                        if not isinstance(layer_states, torch.Tensor):
                            #self.logger.error(f"Unsupported hidden state type for network layer {layer_idx}: {type(layer_states)}")
                            continue
                        
                        #self.logger.info(f"Shape of hidden states for network layer {layer_idx}: {layer_states.shape}")
                    
                        # Extract hidden states of the last token
                        batch_states = []
                        for batch_idx, pos in enumerate(last_token_positions):
                            if batch_idx >= layer_states.shape[0]:
                                continue
                            
                            # Ensure position is within valid range
                            safe_pos = min(pos, layer_states.shape[1] - 1)
                            token_state = layer_states[batch_idx, safe_pos]
                            
                            # Check and handle NaN or Inf
                            if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                                #self.logger.warning(f"Tensor at layer {layer_idx}, batch {batch_idx}, position {safe_pos} contains NaN or Inf, will replace with zero")
                                token_state = torch.zeros_like(token_state)
                                
                            batch_states.append(token_state)
                        
                        if batch_states:
                            # Stack results from all batches
                            stacked_states = torch.stack(batch_states)
                            all_mean_states.append(stacked_states)
                            valid_layers.append(layer_idx)
                            
                            # Record statistics
                            # metadata['mean_stats'][f'layer_{layer_idx}'] = {
                            #     'mean': float(stacked_states.mean().item()),
                            #     'std': float(stacked_states.std().item()),
                            #     'min': float(stacked_states.min().item()),
                            #     'max': float(stacked_states.max().item())
                            # }
                    except Exception as e:
                        #self.logger.error(f"Error processing network layer {layer_idx}: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        continue
        
            if not all_mean_states:
                #self.logger.error("No valid states collected")
                return None
        
            # Stack all network layers
            try:
                stacked_all_layers = torch.stack(all_mean_states)
                
                # Build result
                result = {
                    'hidden_states': stacked_all_layers,  # [num_layers, batch_size, hidden_size]
                    'valid_layers': valid_layers,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
                }
                
                #self.logger.info(f"Final mean hidden_states shape: {result['hidden_states'].shape}")
                #self.logger.info(f"Processed {len(valid_layers)}/{len(valid_layer_indices)} network layers")
                
                return result
            except Exception as e:
                #self.logger.error(f"Error stacking hidden states: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"Error processing output_mean: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_output_eos(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None,
        is_generation_output: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Process hidden state extraction for the output_eos method - simplified version, only extracts the last token of the generated part"""
        try:
            # Validate input
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states is empty")
                return None
    
            if input_ids.numel() == 0:
                #self.logger.error("input_ids is empty")
                return None
        
             # REMOVED: metadata initialization with stats
            # metadata = {
            #     'last_token_stats': {},
            #     'processing_info': []
            # }
              
            # Get input sequence length
            input_length = getattr(self, 'input_length', None)
            if input_length is None:
                #self.logger.warning("Cannot determine input length, using heuristic")
                input_length = max(1, input_ids.shape[1] // 2)  # Assume generated part is half the total length
    
            #self.logger.info(f"Input sequence length: {input_length}, Total sequence length: {input_ids.shape[1]}")
    
            # Handle case where layer_indices is None
            if layer_indices is None:
                if self.use_all_layers:
                    # If using all layers, determine layer count based on hidden state structure
                    if is_generation_output and len(hidden_states) > 0 and isinstance(hidden_states[-1], (tuple, list)):
                        layer_indices = list(range(len(hidden_states[-1])))
                    else:
                        layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # Default to using the first layer
            
            #self.logger.info(f"Using layer indices: {layer_indices}")
        
            # Find the position of the last token in the generated part for each sequence
            last_token_positions = []
            for batch_idx, seq in enumerate(input_ids):
                # Find the last non-padding token in each sequence
                seq_length = (seq != self.tokenizer.pad_token_id).sum().item()
            
                # If sequence length > input length, the last token is in the generated part
                if seq_length > input_length:
                    # Use the last token of the generated part (seq_length - 1)
                    last_token_positions.append(seq_length - 1)
                else:
                    # If sequence length <= input length, there might be no generated part, use the end position of the input
                    #self.logger.warning(f"Sequence {batch_idx} might not have a generated part, using input end position")
                    last_token_positions.append(max(0, seq_length - 1))
            
            #self.logger.info(f"Last token positions of generated part: {last_token_positions}")
        
            # For collecting hidden states
            all_last_token_states = []
            valid_layers = []
        
            # Process hidden states, using different strategies based on whether it's generation model output
            if is_generation_output:
                # For generation model, use hidden states from the last step
                #self.logger.info("Processing generation model hidden states")
            
                # Get hidden states from the last step
                last_step_hidden_states = hidden_states[-1]
            
                # Determine valid layer indices
                if isinstance(last_step_hidden_states, (tuple, list)):
                    available_layers = len(last_step_hidden_states)
                else:
                    #self.logger.error(f"Incorrect type for last step hidden states of generation model: {type(last_step_hidden_states)}")
                    return None
            
                valid_layer_indices = [idx for idx in layer_indices if idx < available_layers]
                if not valid_layer_indices:
                    #self.logger.error(f"No valid layer indices. Requested indices: {layer_indices}, Available layers: {available_layers}")
                    return None
            
                #self.logger.info(f"Will process the following layers: {valid_layer_indices}")
            
                # Process each layer
                for layer_idx in valid_layer_indices:
                    try:
                        layer_states = last_step_hidden_states[layer_idx]
                    
                        # Handle different types of hidden states
                        if isinstance(layer_states, tuple):
                            #self.logger.info(f"Hidden states for layer {layer_idx} are tuple type, attempting to get first element")
                            if len(layer_states) > 0:
                                layer_states = layer_states[0]
                            else:
                                #self.logger.warning(f"Hidden states for layer {layer_idx} is an empty tuple")
                                continue
                        
                        if not isinstance(layer_states, torch.Tensor):
                            #self.logger.error(f"Unsupported hidden state type for layer {layer_idx}: {type(layer_states)}")
                            continue
                        
                        #self.logger.info(f"Shape of hidden states for layer {layer_idx}: {layer_states.shape}")
                    
                        # Extract hidden states of the last token
                        batch_states = []
                        for batch_idx, pos in enumerate(last_token_positions):
                            if batch_idx >= layer_states.shape[0]:
                                continue
                            
                            # Ensure position is within valid range
                            safe_pos = min(pos, layer_states.shape[1] - 1)
                            token_state = layer_states[batch_idx, safe_pos]
                            
                            # Check and handle NaN or Inf
                            if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                                #self.logger.warning(f"Tensor at layer {layer_idx}, batch {batch_idx}, position {safe_pos} contains NaN or Inf, will replace with zero")
                                token_state = torch.zeros_like(token_state)
                                
                            batch_states.append(token_state)
                        
                        if batch_states:
                            # Stack results from all batches
                            stacked_states = torch.stack(batch_states)
                            all_last_token_states.append(stacked_states)
                            valid_layers.append(layer_idx)
                            
                            # Record statistics
                            # metadata['last_token_stats'][f'layer_{layer_idx}'] = {
                            #     'mean': float(stacked_states.mean().item()),
                            #     'std': float(stacked_states.std().item()),
                            #     'min': float(stacked_states.min().item()),
                            #     'max': float(stacked_states.max().item())
                            # }
                    except Exception as e:
                        #self.logger.error(f"Error processing layer {layer_idx}: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        # REMOVED: processing_info append
                        # metadata['processing_info'].append(f"Layer {layer_idx} processing failed: {str(e)}")
                        continue
        
            if not all_last_token_states:
                #self.logger.error("No valid last token hidden states collected")
                return None
        
            # Stack results from all layers and return
            try:
                stacked_all_layers = torch.stack(all_last_token_states)
                
                # Build result
                result = {
                    'hidden_states': stacked_all_layers,  # [num_layers, batch_size, hidden_size]
                    'valid_layers': valid_layers,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids],
                    'positions': last_token_positions
                }
                
                #self.logger.info(f"Final hidden state shape: {result['hidden_states'].shape}")
                #self.logger.info(f"Processed {len(valid_layers)}/{len(valid_layer_indices)} layers")
                
                return result
            except Exception as e:
                #self.logger.error(f"Error building result: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"Error processing output_eos: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def get_hidden_states_and_generate(
        self,
        dataloader: torch.utils.data.DataLoader,
        target_words: List[Tuple[str, str]],
        layer_indices: Optional[List[int]] = None,
        reduce_dims: Optional[int] = None,
        generation_params: Optional[Dict] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Get hidden states and generate responses for the input text.
        
        Args:
            dataloader: DataLoader containing the processed dataset
            target_words: List of target (word, pos) pairs
            layer_indices: Which layers to extract hidden states from
            reduce_dims: Number of dimensions to reduce to (optional)
            generation_params: Parameters for text generation
            batch_size: Batch size for processing
        """
        # Initialize storage
        results = {
            'hidden_states': {f"{word}_{pos}": [] for word, pos in target_words},
            'generations': [],
            'metadata': {
                'input_sentences': [],
                'original_sentences': [],
                'years': [],
                'target_word_positions': {}
            }
        }
        
        # Default generation parameters
        default_gen_params = {
            'max_new_tokens': 50,
            'num_beams': 1,
            'temperature': 0.3,
            'do_sample': True
        }
        generation_params = {**default_gen_params, **(generation_params or {})}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                # Convert input tensors while preserving other fields
                for key in ["input_ids", "attention_mask"]:
                    if key in batch and not isinstance(batch[key], torch.Tensor):
                        batch[key] = torch.tensor(batch[key])
                
                # Get model outputs with hidden states
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Generate responses
                generated_outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    **generation_params,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Process hidden states and generations for each sequence
                for seq_idx in range(batch['input_ids'].shape[0]):
                    # Get tokenized text
                    input_ids = batch['input_ids'][seq_idx]
                    input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    results['metadata']['input_sentences'].append(input_text)
                    
                    # Get and store original sentence and year
                    original_sent = batch['original_sentence'][seq_idx]
                    year_val = batch['Year'][seq_idx]
                    results['metadata']['original_sentences'].append(original_sent)
                    results['metadata']['years'].append(year_val)
                    
                    # Process hidden states for each target word
                    for word, pos in target_words:
                        word_key = f"{word}_{pos}"
                        states = self._extract_hidden_states(
                            outputs.hidden_states,
                            input_ids,
                            method="output_mean"
                        )
                        if states is not None:
                            results['hidden_states'][word_key].append(states['hidden_states'].cpu())
                    
                    # Process generation
                    generated_ids = generated_outputs.sequences[seq_idx]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    generated_text = generated_text.split('assistant:')[-1]
                    results['generations'].append(generated_text)

        # Convert hidden states to tensors and optionally reduce dimensions
        for word_key in results['hidden_states']:
            if results['hidden_states'][word_key]:
                states_tensor = torch.stack(results['hidden_states'][word_key])
                if reduce_dims and reduce_dims < states_tensor.shape[-1]:
                    states_tensor = self._reduce_dimensions(states_tensor, reduce_dims)
                results['hidden_states'][word_key] = states_tensor

        return results

    def _reduce_dimensions(self, hidden_states: torch.Tensor, n_components: int) -> torch.Tensor:
        """Reduce dimensions of hidden states using PCA."""
        original_shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, original_shape[-1])
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(flattened.numpy())
        return torch.tensor(reduced).reshape(*original_shape[:-1], n_components)

    def save_results(self, results: Dict[str, Any], experiment_name: str) -> str:
        """
        Save results to files
        
        Args:
            results: Dictionary containing results
            experiment_name: Experiment name
            
        Returns:
            Path where files were saved
        """
        try:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            self.logger.info(f"Saving results to: {output_dir}")
            
            # If results are a DataFrame, process and save
            if isinstance(results, pd.DataFrame):
                # Create a copy to avoid modifying original data
                df = results.copy()
                
                # Check if there are tensor columns that need separate saving
                tensor_columns = []
                for col in df.columns:
                    if isinstance(df[col].iloc[0], torch.Tensor):
                        tensor_columns.append(col)
                
                # If there are tensor columns, save them separately
                if tensor_columns:
                    self.logger.info(f"Found tensor columns: {tensor_columns}")
                    
                    # Create directory for each tensor column
                    for col in tensor_columns:
                        tensor_dir = os.path.join(output_dir, col)
                        os.makedirs(tensor_dir, exist_ok=True)
                        
                        # Save each tensor
                        for idx, row in df.iterrows():
                            tensor = row[col]
                            tensor_path = os.path.join(tensor_dir, f"{idx}.pt")
                            torch.save(tensor, tensor_path)
                            self.logger.debug(f"Saving tensor to: {tensor_path}")
                        
                        # Remove tensor columns from DataFrame
                        df = df.drop(columns=[col])
                
                # Save the processed DataFrame
                csv_path = os.path.join(output_dir, f"{experiment_name}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved CSV to: {csv_path}")
                
            # If results are a dictionary, save as JSON
            elif isinstance(results, dict):
                # Process tensors in the dictionary
                processed_results = {}
                
                for key, value in results.items():
                    if isinstance(value, torch.Tensor):
                        # Save tensor to file
                        tensor_path = os.path.join(output_dir, f"{key}.pt")
                        torch.save(value, tensor_path)
                        self.logger.info(f"Saved tensor to: {tensor_path}")
                        
                        # Record path in the results dictionary
                        processed_results[key] = f"tensor_saved_at:{tensor_path}"
                    else:
                        processed_results[key] = value
                
                # Save the processed dictionary as JSON
                json_path = os.path.join(output_dir, f"{experiment_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_results, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"Saved JSON to: {json_path}")
                
            else:
                error_msg = f"Unsupported result type: {type(results)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            return output_dir
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def analyze_results(
        self,
        results: Dict[str, Any],
        n_clusters: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze inference results.
        
        Args:
            results: Dictionary containing inference results
            n_clusters: Number of clusters for analysis
        """
        from sklearn.cluster import KMeans
        
        analysis = {}
        
        # Analyze hidden states
        for word_key, states in results['hidden_states'].items():
            if len(states) == 0:
                continue
                
            states_2d = states.reshape(states.shape[0], -1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(states_2d)
            
            analysis[word_key] = {
                'clusters': clusters,
                'centroids': kmeans.cluster_centers_,
                'cluster_sizes': np.bincount(clusters).tolist()
            }
        
        return analysis

    def save_consolidated_hidden_states(self, results_df: pd.DataFrame, experiment_name: str):
        """
        Combine and save all hidden states into a single tensor file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Check tensor columns
        tensor_columns = []
        for col in results_df.columns:
            if len(results_df) > 0 and isinstance(results_df[col].iloc[0], torch.Tensor):
                tensor_columns.append(col)
        
        self.logger.info(f"Found {len(tensor_columns)} tensor columns: {tensor_columns}")
        
        # Process each tensor column
        for col in tensor_columns:
            try:
                # Stack all tensors in the column
                tensors = [tensor for tensor in results_df[col] if tensor is not None]
                if not tensors:
                    self.logger.warning(f"No valid tensors found in column {col}")
                    continue
                    
                stacked_tensor = torch.stack(tensors)
                
                # Save the combined tensor
                file_name = f"hidden_states_{col.replace('/', '_')}.pt"
                file_path = os.path.join(output_dir, file_name)
                torch.save(stacked_tensor, file_path)
                self.logger.info(f"Saved combined hidden states to: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error saving combined hidden states for {col}: {e}")
                self.logger.error(traceback.format_exc())
