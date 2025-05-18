from datasets import Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import ast
import re
from typing import Optional, Dict, List, Union, Any, Tuple
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from langdetect import detect_langs
from pandarallel import pandarallel
import traceback

# Import custom log configuration
from logger_config import setup_logger, TRACE

# Set module-specific logger
logger = setup_logger('dataloader', 'logs/dataloader.log')

def unify_target_casing(sentence: str, target_word: str) -> str:
    """
    Replace occurrences of target_word (case-insensitive) in sentence with the lowercase form of target_word.
    For example: if target_word="bank", replace "Bank", "BANK", "bAnK", etc., with "bank".
    """
    pattern = re.compile(r'(?i)\b' + re.escape(target_word) + r'\b')
    return pattern.sub(target_word.lower(), sentence)


# Filtering logic
def is_english(text: str, threshold: float = 0.7) -> bool:
    """Check if the text is English"""
    try:
        languages = detect_langs(text)
        en_prob = next((lang.prob for lang in languages if lang.lang == 'en'), 0.0)
        return en_prob >= threshold
    except:
        return False

def is_valid_sentence(
    sentence: str, 
    min_token_length: int = 5,
    max_token_length: int = 256,
    max_consecutive_digits: int = 6,
    allowed_end_punct: tuple = ('.', '?', '!', ';', '-', ':', '"'),
    max_non_english_chars: int = 3,
    min_alpha_chars: int = 5
) -> bool:
    """Check if the sentence meets filtering criteria"""
    # Quick fail condition: empty string
    if not sentence.strip():
        return False
    if sentence[0] in '0123456789+-)':
        return False
    # 1. Check ending punctuation (if enabled)
    if allowed_end_punct is not None:
        if len(sentence) == 0 or sentence[-1] not in allowed_end_punct:
            return False
            
    # 2. Check email and URL formats (enhanced)
    if re.search(r'\S+@\S+\.\S+', sentence, re.IGNORECASE):  # Email
        return False
    # Enhanced URL detection, supports more TLDs
    if re.search(r'\bwww\.\S+\.[a-z]{2,}', sentence, re.IGNORECASE):
        return False
    # Detect URL format (with or without www)
    if re.search(r'https?://\S+', sentence, re.IGNORECASE):
        return False
        
    # 3. Check phone numbers and consecutive digits
    # Detect common phone number formats
    if re.search(r'(?:\+\d{1,4}[-\s]?)?\(?\d{1,4}\)?[-\s]?\d{1,4}[-\s]?\d{1,4}', sentence):
        return False
    # Detect consecutive digits (ignore spaces)
    sentence_no_space = sentence.replace(' ', '')
    if re.search(r'\d{{{},}}'.format(max_consecutive_digits+1), sentence_no_space):
        return False
        
    # 4. Check length by splitting by space
    tokens = sentence.split()
    if not (min_token_length <= len(tokens) <= max_token_length):
        return False
                        
    # 5. Check number of non-English characters
    non_english_count = sum(1 for c in sentence if ord(c) > 127)
    if non_english_count > max_non_english_chars:
        return False
        
    # 6. Check minimum number of alphabetic characters
    alpha_count = sum(c.isalpha() for c in sentence)
    if alpha_count < min_alpha_chars:
        return False
    
    # 7. Check if it is English
    if not is_english(sentence):
        return False
        
    return True


class CustomDataLoader:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        target_words: List[Tuple[str, str]],  # List of (word, pos) tuples
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 16,
        prompt_template: Optional[List[Dict[str, str]]] = None,
        perplexity_threshold: float = 9.8,  # Default log perplexity threshold
        context_mode: str = "none",   # "none", "sentence", "token"
        context_window: int = 0,      # how  many sentences or tokens to add on each side
        simple_filter: bool = True,   # Whether to apply simple filtering rules
        duplicate_handling: str = "remove",  # New parameter: ["mask", "remove"]
        min_sentence_length: int = 5,  # Minimum sentence length
        max_sentence_length: int = 100,  # Maximum sentence length
        max_samples: Optional[int] = None,  # Maximum number of samples
        log_file: str = "logs/dataloader.log"  # Log file path
    ):
        """
        Initialize the CustomDataLoader.
        """
        # Set module-specific logger
        self.logger = setup_logger('dataloader', log_file)
        
        self.tokenizer = tokenizer
        self.target_words = target_words
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.perplexity_threshold = perplexity_threshold
        
        # New context arguments
        self.context_mode = context_mode
        self.context_window = context_window
        self.simple_filter = simple_filter
        self.duplicate_handling = duplicate_handling
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.max_samples = max_samples
        
        # Initialize prompt template
        self.prompt_template = prompt_template
        
        # Initialize pandarallel for parallel processing
        try:
            pandarallel.initialize(progress_bar=True, nb_workers=self.num_workers)
            self.logger.info(f"Initialized pandarallel, using {self.num_workers} workers")
        except Exception as e:
            self.logger.warning(f"Failed to initialize pandarallel: {e}, will use serial processing")
        
        self.logger.info(f"DataLoader configuration: Target words: {target_words}, Batch size: {batch_size}, Max length: {max_length}, "
                         f"Context mode: {context_mode}, Context window: {context_window}, Simple filter: {simple_filter}, "
                         f"Duplicate handling: {duplicate_handling}, Sentence length: {min_sentence_length}-{max_sentence_length}")
        
        self.target_words_set = {}
        for word, pos in target_words:
            lower_word = word.lower()
            # For noun POS tags (assuming they start with 'N'), store original word and substring matching flag
            if pos.lower().startswith('n'):
                self.target_words_set[(lower_word, pos)] = True  # Mark as needing substring matching
            else:
                self.target_words_set[(lower_word, pos)] = False  # Exact match

        # Parameter validation
        if self.duplicate_handling not in ["mask", "remove"]:
            raise ValueError("duplicate_handling must be 'mask' or 'remove'")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_duplicate(self, tokens: List[str], target_indices: dict) -> List[str]:
        """Modify token list based on handling mode"""
        if self.duplicate_handling == "mask":
            return [t if i not in target_indices else "[MASK]" for i, t in enumerate(tokens)]
        elif self.duplicate_handling == "remove":
            return [t for i, t in enumerate(tokens) if i not in target_indices]
        return tokens

    def apply_simple_filter(self, df: pd.DataFrame, **filter_params) -> pd.DataFrame:
        """
        Apply several simple filtering rules to the 'sentence' column of df
        """
        # Apply filtering using parallel processing
        mask = df['sentence'].parallel_apply(lambda s: is_valid_sentence(s, **filter_params))
        return df[mask].copy()

    def set_filtering_params(self, perplexity_threshold: float = None):
        """
        Update filtering parameters.
        """
        if perplexity_threshold is not None:
            self.perplexity_threshold = perplexity_threshold

    def filter_sentences_by_word_pos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parallelized target word filtering logic"""
        self.logger.info(f"Starting sentence filtering, total {len(df)} sentences")
        
        def process_row(row):
            try:
                sentence = row['sentence']
                
                # Check sentence length
                if len(sentence) < self.min_sentence_length or len(sentence) > self.max_sentence_length:
                    return None
                
                # Check each target word
                for target_word, target_pos in self.target_words:
                    # Check if target word is in sentence (case-insensitive)
                    pattern = re.compile(r'(?i)\b' + re.escape(target_word) + r'\b')
                    if pattern.search(sentence):
                        # Unify target word casing
                        sentence = unify_target_casing(sentence, target_word)
                        
                        # Create new row
                        new_row = row.copy()
                        new_row['sentence'] = sentence
                        new_row['target_word'] = target_word
                        new_row['target_pos'] = target_pos
                        
                        return new_row
            except Exception:
                return None
            return None
        
        # Use parallel processing
        results = df.parallel_apply(process_row, axis=1)
        filtered_rows = [row for row in results if row is not None]
        
        result_df = pd.DataFrame(filtered_rows)
        self.logger.info(f"Filtering complete, total {len(result_df)} results")
        
        # If max_samples is set, perform random sampling
        if self.max_samples and len(result_df) > self.max_samples:
            result_df = result_df.sample(n=self.max_samples, random_state=42)
            self.logger.info(f"After random sampling, total {len(result_df)} results")
        
        return result_df

    def expand_context_by_sentences(self, df: pd.DataFrame, separator: str = " ") -> pd.DataFrame:
        """
        Parallelized context expansion method
        """
        if self.context_window <= 0:
            df['original_sentence'] = df['sentence']
            return df

        df = df.sort_values(by=['MISC']).reset_index(drop=True)
        
        def process_group(group_data):
            sentences = group_data['sentence'].tolist()
            target_words = group_data['target_word'].tolist()  
            target_pos_list = group_data['target_pos'].tolist()
            misc_val = group_data['MISC'].iloc[0]
            
            results = []
            for i in range(len(sentences)):
                # Collect context window
                start = max(0, i - self.context_window)
                end = min(len(sentences), i + self.context_window + 1)
                
                # Extract the original sentence and context sentences separately
                original_sentence = sentences[i]
                context_sentences = sentences[start:i] + sentences[i+1:end]
                
                # Process context sentences to handle duplicates of target word
                processed_context = []
                current_target = target_words[i].lower()
                
                for context_sent in context_sentences:
                    tokens = context_sent.split()
                    target_indices = {}
                    
                    # Find target words in context sentence
                    for j, token in enumerate(tokens):
                        current_word = token.lower()
                        # If is noun, use substring matching
                        if target_pos_list[i].startswith('N'):
                            if current_target in current_word:
                                target_indices[j] = True
                        # Other POS use exact matching
                        else:
                            if current_word == current_target:
                                target_indices[j] = True
                    
                    # Process tokens based on duplicate_handling setting
                    if self.duplicate_handling == "mask":
                        tokens = [t if j not in target_indices else "[MASK]" for j, t in enumerate(tokens)]
                    elif self.duplicate_handling == "remove":
                        tokens = [t for j, t in enumerate(tokens) if j not in target_indices]
                    
                    processed_context.append(" ".join(tokens))
                
                # Combine original sentence with processed context sentences
                all_sentences = processed_context[:i-start] + [original_sentence] + processed_context[i-start:]
                processed_sentence = separator.join(all_sentences).replace("  ", " ")
                
                results.append({
                    'sentence': processed_sentence,
                    'original_sentence': original_sentence,
                    'MISC': misc_val,
                    'Year': group_data['Year'].iloc[i],
                    'target_word': target_words[i],  
                    'target_pos': target_pos_list[i]
                })
            return results
        
        # Process each group in parallel
        all_results = []
        with mp.Pool(processes=self.num_workers) as pool:
            group_results = pool.map(process_group, [group for _, group in df.groupby('MISC')])
            for result in group_results:
                all_results.extend(result)
        
        return pd.DataFrame(all_results)

    def expand_context_by_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conceptual method to expand context by tokens. 
        (Placeholder: no real token-level expansion here)
        """
        if self.context_window <= 0:
            return df
            
        # Save original sentences before any expansion
        df['original_sentence'] = df['sentence']
        return df

    def format_prompt(
        self, 
        sentence: str, 
        word: str, 
        pos: str
    ) -> List[Dict[str, str]]:
        """
        Generate dialogue format based on preset template
        Template variables supported: {sentence}, {word}, {pos}
        """
        formatted_messages = []
        
        for message in self.prompt_template:
            # Deep copy message template to avoid modifying original data
            formatted_message = message.copy()
            
            # Only handle content replacement for user and system roles
            if message['role'] in ['system', 'user']:
                formatted_content = message['content'].format(
                    sentence=sentence,
                    word=word,
                    pos=pos
                )
                formatted_message['content'] = formatted_content
            
            formatted_messages.append(formatted_message)
        
        return formatted_messages

    def load_dataset(self, 
                    data_paths: Union[str, List[str]]) -> Dataset:
        """
        Load and prepare dataset from one or multiple CSV files.
        """
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        dfs = []
        for path in data_paths:
            try:
                df = pd.read_csv(path)
                self.logger.info(f"Loaded {len(df)} sentences from {path}")
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error loading CSV file {path}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid CSV files were loaded")
        
        df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Total sentences after combining files: {len(df)}")
        
        # Remove duplicates if any
        original_len = len(df)
        df = df.drop_duplicates(subset=['sentence'], keep='first')
        if len(df) < original_len:
            self.logger.info(f"Removed {original_len - len(df)} duplicate sentences")
            
        # Unify pos_tags format conversion
        def _convert_pos_tags(pos_str):
            try:
                return ast.literal_eval(pos_str) if isinstance(pos_str, str) else pos_str
            except:
                return []  # If conversion fails, return an empty list
        
        if 'pos_tags' in df.columns:
            # Use pandarallel for acceleration (ensure initialized)
            df['pos_tags'] = df['pos_tags'].parallel_apply(_convert_pos_tags)  # Use pandarallel for acceleration
            self.logger.info("Converted pos_tags to list of tuples")
        
        # Apply filtering based on perplexity
        if 'perplexity' not in df.columns:
            raise ValueError("Perplexity scores not found in the dataset. Run preprocessing first.")
        
        original_len = len(df)
        filter_condition = np.log(df['perplexity']) < self.perplexity_threshold
        df = df[filter_condition]
        filtered_len = len(df)
        self.logger.info(f"Filtered {original_len - filtered_len} sentences that don't meet filtering criteria")
        self.logger.info(f"Perplexity threshold: {self.perplexity_threshold}")
        
        if filtered_len == 0:
            raise ValueError("No sentences found meeting the filtering criteria")

        # Apply simple filtering if enabled
        if self.simple_filter:
            before_simple_len = len(df)
            df = self.apply_simple_filter(df)
            self.logger.info(f"Simple filter removed {before_simple_len - len(df)} sentences.")
        
        # First perform target word and POS filtering
        filtered_df = self.filter_sentences_by_word_pos(df)
        
        if len(filtered_df) == 0:
            raise ValueError("No sentences found matching the target words and POS tags")
        
        self.logger.info(f"Found {len(filtered_df)} sentences matching all criteria")

        # Add context after filtering (no longer need to handle POS tags at this point)
        if self.context_mode == "sentence":
            filtered_df = self.expand_context_by_sentences(filtered_df)
        elif self.context_mode == "token":
            filtered_df = self.expand_context_by_tokens(filtered_df)

        # Add check after generating the final dataset
        self.logger.info("Final dataset columns:", filtered_df.columns.tolist())
        # Expected output should not include target_token_index and target_occurrence

        dataset = Dataset.from_pandas(filtered_df)
        
        # Use map function for preprocessing
        dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing dataset"
        )
        
        self.logger.info("\nFirst 3 dataset examples:")
        for i in range(min(3, len(dataset))):
            self.logger.info(f"Example {i+1}: {dataset[i]}")
        
        return dataset
    def preprocess_function(self, examples: Dict) -> Dict:
        """Add field existence check"""
        # New field check
        required_fields = ['sentence', 'target_word', 'target_pos', 'Year']
        for field in required_fields:
            if field not in examples:
                raise ValueError(f"Missing required field '{field}' in dataset examples")
        
        model_inputs = []
        
        for i in range(len(examples['sentence'])):
            # Only pass necessary parameters
            conversation = self.format_prompt(
                sentence=examples['sentence'][i],
                word=examples['target_word'][i],
                pos=examples['target_pos'][i]
            )
            
            # If tokenizer has a custom chat template method
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_input = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Simply join into a string
                formatted_input = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation
                ])
            
            model_inputs.append(formatted_input)
        
        # Tokenize
        tokenized = self.tokenizer(
            model_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Keep original information (remove unnecessary fields)
        tokenized['sentence'] = examples['sentence']
        tokenized['target_word'] = examples['target_word']
        tokenized['target_pos'] = examples['target_pos']
        tokenized['Year'] = examples['Year']  # Keep Year information
        tokenized['original_sentence'] = examples.get('original_sentence', examples['sentence'])
        
        return tokenized

    def get_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from dataset."""
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

        def custom_collate_fn(features):
            text_fields = {
                "sentence": [],
                "target_word": [],
                "target_pos": [],
                "Year": [],  # Keep necessary fields
                "original_sentence": [],
            }
            tensor_input = []

            for f in features:
                # Extract valid fields
                text_fields["sentence"].append(f.pop("sentence"))
                text_fields["target_word"].append(f.pop("target_word"))
                text_fields["target_pos"].append(f.pop("target_pos"))
                text_fields["Year"].append(f.pop("Year"))
                text_fields["original_sentence"].append(f.pop("original_sentence", f.get("sentence")))
                
                tensor_input.append(f)

            batch = data_collator(tensor_input)

            # Add text fields back
            batch.update(text_fields)
            
            return batch
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn
        )        

    def load_and_process_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and process data

        Args:
            data_path: Path to the data file

        Returns:
            Processed DataFrame
        """
        try:
            self.logger.info(f"Starting data loading: {data_path}")
            
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path, lines=True)
            elif data_path.endswith('.jsonl'):
                df = pd.read_json(data_path, lines=True)
            else:
                self.logger.error(f"Unsupported file format: {data_path}")
                return None
            
            self.logger.info(f"Original data size: {len(df)}")
            
            # Ensure there is a sentence column
            if 'sentence' not in df.columns and 'text' in df.columns:
                df['sentence'] = df['text']
                self.logger.info("Renamed 'text' column to 'sentence'")
            
            if 'sentence' not in df.columns:
                self.logger.error("No 'sentence' or 'text' column in data")
                return None
            
            # Apply simple filtering rules
            if self.simple_filter:
                original_size = len(df)
                df = self.apply_simple_filter(df)
                self.logger.info(f"After simple filtering, data size reduced from {original_size} to {len(df)}")
            
            # Filter sentences containing target words
            df = self.filter_sentences_by_word_pos(df)
            
            # Expand context based on context mode
            if self.context_mode == "sentence" and self.context_window > 0:
                self.logger.info(f"Expanding context using sentence mode, window size: {self.context_window}")
                df = self.expand_context_by_sentences(df)
            elif self.context_mode == "token" and self.context_window > 0:
                self.logger.info(f"Expanding context using token mode, window size: {self.context_window}")
                df = self.expand_context_by_tokens(df)
            
            self.logger.info(f"Processing complete, final data size: {len(df)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading and processing data: {e}")
            self.logger.error(traceback.format_exc())
            return None
