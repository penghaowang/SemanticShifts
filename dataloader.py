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

# 导入自定义日志配置
from logger_config import setup_logger, TRACE

# 设置模块特定的日志记录器
logger = setup_logger('dataloader', 'logs/dataloader.log')

def unify_target_casing(sentence: str, target_word: str) -> str: 
    """
    将 sentence 中出现的 target_word（忽略大小写）替换为 target_word 的小写形式。
    例如：如果 target_word="bank", 则把 "Bank", "BANK", "bAnK" 等都替换成 "bank"。
    """
    pattern = re.compile(r'(?i)\b' + re.escape(target_word) + r'\b')
    return pattern.sub(target_word.lower(), sentence)


# 提取的过滤逻辑
def is_english(text: str, threshold: float = 0.7) -> bool:
    """检查文本是否为英语"""
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
    """检查句子是否符合过滤条件"""
    # 快速失败条件：空字符串
    if not sentence.strip():
        return False
    if sentence[0] in '0123456789+-)':
        return False
    # 1. 检查结尾标点（如果启用）
    if allowed_end_punct is not None:
        if len(sentence) == 0 or sentence[-1] not in allowed_end_punct:
            return False
            
    # 2. 检查邮箱和网址格式（增强版）
    if re.search(r'\S+@\S+\.\S+', sentence, re.IGNORECASE):  # 邮箱
        return False
    # 增强网址检测，支持更多顶级域名
    if re.search(r'\bwww\.\S+\.[a-z]{2,}', sentence, re.IGNORECASE):
        return False
    # 检测URL格式（带或不带www）
    if re.search(r'https?://\S+', sentence, re.IGNORECASE):
        return False
        
    # 3. 检查电话号码和连续数字
    # 检测常见电话号码格式
    if re.search(r'(?:\+\d{1,4}[-\s]?)?\(?\d{1,4}\)?[-\s]?\d{1,4}[-\s]?\d{1,4}', sentence):
        return False
    # 检测连续数字（忽略空格）
    sentence_no_space = sentence.replace(' ', '')
    if re.search(r'\d{{{},}}'.format(max_consecutive_digits+1), sentence_no_space):
        return False
        
    # 4. 按空格分割检查长度
    tokens = sentence.split()
    if not (min_token_length <= len(tokens) <= max_token_length):
        return False
                        
    # 5. 检查非英文字符数
    non_english_count = sum(1 for c in sentence if ord(c) > 127)
    if non_english_count > max_non_english_chars:
        return False
        
    # 6. 检查最小英文字母数
    alpha_count = sum(c.isalpha() for c in sentence)
    if alpha_count < min_alpha_chars:
        return False
    
    # 7. 检查是否为英文
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
        duplicate_handling: str = "remove",  # 新增参数：["mask", "remove"]
        min_sentence_length: int = 5,  # 最小句子长度
        max_sentence_length: int = 100,  # 最大句子长度
        max_samples: Optional[int] = None,  # 最大样本数量
        log_file: str = "logs/dataloader.log"  # 日志文件路径
    ):
        """
        Initialize the CustomDataLoader.
        """
        # 设置模块特定的日志记录器
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
            self.logger.info(f"已初始化pandarallel，使用{self.num_workers}个工作进程")
        except Exception as e:
            self.logger.warning(f"初始化pandarallel失败: {e}，将使用串行处理")
        
        self.logger.info(f"数据加载器配置: 目标词: {target_words}, 批处理大小: {batch_size}, 最大长度: {max_length}, "
                         f"上下文模式: {context_mode}, 上下文窗口: {context_window}, 简单过滤: {simple_filter}, "
                         f"重复处理: {duplicate_handling}, 句子长度: {min_sentence_length}-{max_sentence_length}")
        
        self.target_words_set = {}
        for word, pos in target_words:
            lower_word = word.lower()
            # 对名词类POS标签（假设以'N'开头）存储原始词和子字符串匹配标记
            if pos.lower().startswith('n'):
                self.target_words_set[(lower_word, pos)] = True  # 标记需要子字符串匹配
            else:
                self.target_words_set[(lower_word, pos)] = False  # 精确匹配

        # 参数验证
        if self.duplicate_handling not in ["mask", "remove"]:
            raise ValueError("duplicate_handling must be 'mask' or 'remove'")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_duplicate(self, tokens: List[str], target_indices: dict) -> List[str]:
        """根据处理模式修改token列表"""
        if self.duplicate_handling == "mask":
            return [t if i not in target_indices else "[MASK]" for i, t in enumerate(tokens)]
        elif self.duplicate_handling == "remove":
            return [t for i, t in enumerate(tokens) if i not in target_indices]
        return tokens

    def apply_simple_filter(self, df: pd.DataFrame, **filter_params) -> pd.DataFrame:
        """
        对 df 中的 'sentence' 列执行若干简单规则过滤
        """
        # 使用并行处理应用过滤
        mask = df['sentence'].parallel_apply(lambda s: is_valid_sentence(s, **filter_params))
        return df[mask].copy()

    def set_filtering_params(self, perplexity_threshold: float = None):
        """
        Update filtering parameters.
        """
        if perplexity_threshold is not None:
            self.perplexity_threshold = perplexity_threshold

    def filter_sentences_by_word_pos(self, df: pd.DataFrame) -> pd.DataFrame:
        """并行化的目标词筛选逻辑"""
        self.logger.info(f"开始筛选句子，共 {len(df)} 条")
        
        def process_row(row):
            try:
                sentence = row['sentence']
                
                # 检查句子长度
                if len(sentence) < self.min_sentence_length or len(sentence) > self.max_sentence_length:
                    return None
                
                # 检查每个目标词
                for target_word, target_pos in self.target_words:
                    # 忽略大小写检查目标词是否在句子中
                    pattern = re.compile(r'(?i)\b' + re.escape(target_word) + r'\b')
                    if pattern.search(sentence):
                        # 统一目标词大小写
                        sentence = unify_target_casing(sentence, target_word)
                        
                        # 创建新行
                        new_row = row.copy()
                        new_row['sentence'] = sentence
                        new_row['target_word'] = target_word
                        new_row['target_pos'] = target_pos
                        
                        return new_row
            except Exception:
                return None
            return None
        
        # 使用并行处理
        results = df.parallel_apply(process_row, axis=1)
        filtered_rows = [row for row in results if row is not None]
        
        result_df = pd.DataFrame(filtered_rows)
        self.logger.info(f"筛选完成，结果共 {len(result_df)} 条")
        
        # 如果设置了最大样本数量，随机抽样
        if self.max_samples and len(result_df) > self.max_samples:
            result_df = result_df.sample(n=self.max_samples, random_state=42)
            self.logger.info(f"随机抽样后，结果共 {len(result_df)} 条")
        
        return result_df

    def expand_context_by_sentences(self, df: pd.DataFrame, separator: str = " ") -> pd.DataFrame:
        """
        并行化的上下文扩展方法
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
        
        # 并行处理每个组
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
        根据预设模板生成对话格式
        模板变量支持：{sentence}, {word}, {pos}
        """
        formatted_messages = []
        
        for message in self.prompt_template:
            # 深拷贝消息模板避免修改原始数据
            formatted_message = message.copy()
            
            # 仅处理用户和系统角色的内容替换
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
                    data_paths: Union[str, List[str]]:
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
            
        # 统一转换 pos_tags 格式
        def _convert_pos_tags(pos_str):
            try:
                return ast.literal_eval(pos_str) if isinstance(pos_str, str) else pos_str
            except:
                return []  # 如果转换失败，返回空列表
        
        if 'pos_tags' in df.columns:
            # 使用 pandarallel 加速（确保已初始化）
            df['pos_tags'] = df['pos_tags'].parallel_apply(_convert_pos_tags)  # 使用 pandarallel 加速
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
        
        # 先进行目标词和POS的筛选
        filtered_df = self.filter_sentences_by_word_pos(df)
        
        if len(filtered_df) == 0:
            raise ValueError("No sentences found matching the target words and POS tags")
        
        self.logger.info(f"Found {len(filtered_df)} sentences matching all criteria")

        # 在筛选之后添加上下文（此时不再需要处理POS标签）
        if self.context_mode == "sentence":
            filtered_df = self.expand_context_by_sentences(filtered_df)
        elif self.context_mode == "token":
            filtered_df = self.expand_context_by_tokens(filtered_df)

        # 在生成最终数据集后添加检查
        self.logger.info("Final dataset columns:", filtered_df.columns.tolist())
        # 预期输出应不包含 target_token_index 和 target_occurrence

        dataset = Dataset.from_pandas(filtered_df)
        
        # 用 map 函数做预处理
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
        """增加字段存在性检查"""
        # 新增字段检查
        required_fields = ['sentence', 'target_word', 'target_pos', 'Year']
        for field in required_fields:
            if field not in examples:
                raise ValueError(f"Missing required field '{field}' in dataset examples")
        
        model_inputs = []
        
        for i in range(len(examples['sentence'])):
            # 仅传递必要参数
            conversation = self.format_prompt(
                sentence=examples['sentence'][i],
                word=examples['target_word'][i],
                pos=examples['target_pos'][i]
            )
            
            # 如果 tokenizer 带有自定义的 chat 模板方法
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_input = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 简单拼成字符串
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
        
        # 保留原信息（移除非必要字段）
        tokenized['sentence'] = examples['sentence']
        tokenized['target_word'] = examples['target_word']
        tokenized['target_pos'] = examples['target_pos']
        tokenized['Year'] = examples['Year']  # 保留Year信息
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
                "Year": [],  # 保留必要字段
                "original_sentence": [],
            }
            tensor_input = []

            for f in features:
                # 提取有效字段
                text_fields["sentence"].append(f.pop("sentence"))
                text_fields["target_word"].append(f.pop("target_word"))
                text_fields["target_pos"].append(f.pop("target_pos"))
                text_fields["Year"].append(f.pop("Year"))
                text_fields["original_sentence"].append(f.pop("original_sentence", f.get("sentence")))
                
                tensor_input.append(f)

            batch = data_collator(tensor_input)

            # 添加回文本字段
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
        加载并处理数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            处理后的DataFrame
        """
        try:
            self.logger.info(f"开始加载数据: {data_path}")
            
            # 加载数据
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path, lines=True)
            elif data_path.endswith('.jsonl'):
                df = pd.read_json(data_path, lines=True)
            else:
                self.logger.error(f"不支持的文件格式: {data_path}")
                return None
            
            self.logger.info(f"原始数据大小: {len(df)}")
            
            # 确保有sentence列
            if 'sentence' not in df.columns and 'text' in df.columns:
                df['sentence'] = df['text']
                self.logger.info("将'text'列重命名为'sentence'")
            
            if 'sentence' not in df.columns:
                self.logger.error("数据中没有'sentence'或'text'列")
                return None
            
            # 应用简单过滤规则
            if self.simple_filter:
                original_size = len(df)
                df = self.apply_simple_filter(df)
                self.logger.info(f"简单过滤后，数据大小从 {original_size} 减少到 {len(df)}")
            
            # 筛选包含目标词的句子
            df = self.filter_sentences_by_word_pos(df)
            
            # 根据上下文模式扩展上下文
            if self.context_mode == "sentence" and self.context_window > 0:
                self.logger.info(f"使用句子模式扩展上下文，窗口大小: {self.context_window}")
                df = self.expand_context_by_sentences(df)
            elif self.context_mode == "token" and self.context_window > 0:
                self.logger.info(f"使用token模式扩展上下文，窗口大小: {self.context_window}")
                df = self.expand_context_by_tokens(df)
            
            self.logger.info(f"处理完成，最终数据大小: {len(df)}")
            return df
            
        except Exception as e:
            self.logger.error(f"加载和处理数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
