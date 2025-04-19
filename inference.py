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

# 导入自定义日志配置
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
        extraction_method: str = "input_last_token",  # 可以是单个方法或"all"
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
        初始化模型推理类
        
        Args:
            model_name: 模型名称或路径
            layer_indices: 需要提取的层索引列表，None表示所有层
            batch_size: 批处理大小
            num_beams: beam search的beam数量
            temperature: 生成温度
            max_new_tokens: 最大生成token数
            prompt_template: prompt模板列表
            extraction_method: hidden states提取方法
                - input_last_token: 提取输入的最后一个token
                - eos_token: 在输入后添加EOS token并提取
                - input_mean: 对输入句子所有token做mean pooling
                - output_mean: 对生成内容做mean pooling
                - output_eos: 提取生成内容的EOS token
                - all: 提取所有方法的hidden states（优化计算）
            use_all_layers: 是否使用所有层的hidden states
            target_word: Target word for extraction
            target_pos: Target position for extraction
            saved_dataset_dir: 保存的数据集目录路径
            output_dir: 输出目录路径
            save_hidden_states: 是否保存隐藏状态
            log_file: 日志文件路径
            use_8bit: 是否使用8位量化
        """
        # 设置模块特定的日志记录器
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
        
        # 初始化模型和tokenizer
        self.logger.info(f"正在加载模型: {model_name}，使用8位量化: {use_8bit}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            load_in_8bit=use_8bit
        )
        
        # 设置pad_token_id和eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.debug("设置pad_token_id为eos_token_id")
            
        # 记录模型配置
        self.logger.info("=== 模型配置 ===")
        self.logger.info(f"  模型名称: {model_name}")
        self.logger.info(f"  层索引: {layer_indices}")
        self.logger.info(f"  批处理大小: {batch_size}")
        self.logger.info(f"  Beam数量: {num_beams}")
        self.logger.info(f"  生成温度: {temperature}")
        self.logger.info(f"  最大生成token数: {max_new_tokens}")
        self.logger.info(f"  提取方法: {extraction_method}")
        self.logger.info(f"  使用所有层: {use_all_layers}")
        self.logger.info(f"  目标词: {target_word}")
        self.logger.info(f"  目标位置: {target_pos}")
        if saved_dataset_dir:
            self.logger.info(f"  保存的数据集目录: {saved_dataset_dir}")
        
        # 新增层数验证逻辑
        self.total_layers = self.model.config.num_hidden_layers
        if layer_indices and any(idx >= self.total_layers for idx in self.layer_indices):
            error_msg = f"层索引超出模型总层数 ({self.total_layers})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"模型初始化完成，总层数: {self.total_layers}")
        
    def format_prompt(self, sentence: str, word: str) -> List[Dict[str, str]]:
        """
        根据模板格式化prompt
        
        Args:
            sentence: 输入句子
            word: 目标词
            
        Returns:
            格式化后的prompt列表
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
        """从保存的目录加载数据集"""
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
        对输入数据运行推理
        
        Args:
            df: 包含sentence列的DataFrame，如果为None则尝试从保存的数据集加载
            
        Returns:
            包含推理结果的DataFrame
        """
        # 如果没有提供DataFrame，尝试加载保存的数据集
        if df is None:
            dataset = self.load_saved_dataset()
            if dataset is None:
                raise ValueError("No input data provided and failed to load saved dataset")
            df = pd.DataFrame(dataset)
        
        results = []
        all_hidden_states = {}  # 用于按方法收集隐藏状态的字典
        
        # 按批次处理数据
        for i in tqdm(range(0, len(df), self.batch_size)):
            batch_df = df.iloc[i:i + self.batch_size]
            
            # 准备批次数据
            batch_prompts = []
            for _, row in batch_df.iterrows():
                prompt = self.format_prompt(
                    row['sentence'], 
                    self.target_word if self.target_word else row.get('word', '')
                )
                batch_prompts.append(prompt)
            
            # 获取隐藏状态和生成结果
            batch_results = self._process_batch(batch_prompts, batch_df)
            
            # 提取并收集隐藏状态
            for result in batch_results:
                for key, value in list(result.items()):
                    if isinstance(value, torch.Tensor):
                        # 记录张量形状以便调试
                        #self.logger.info(f"收集张量 {key}，形状: {value.shape}")
                        
                        if key not in all_hidden_states:
                            all_hidden_states[key] = []
                        
                        # 确保张量在CPU上，并且是浮点类型
                        try:
                            tensor_to_save = value.detach().cpu().float()
                            
                            # 检查张量是否有效
                            if torch.isnan(tensor_to_save).any() or torch.isinf(tensor_to_save).any():
                                #self.logger.warning(f"张量 {key} 包含NaN或Inf值，将被替换为零张量")
                                tensor_to_save = torch.zeros_like(tensor_to_save)
                                
                            all_hidden_states[key].append(tensor_to_save)
                            
                            # 从结果中移除张量以避免内存问题
                            result[key] = f"tensor_collected_for_{key}"
                        except Exception as e:
                            self.logger.error(f"处理张量 {key} 时出错: {str(e)}")
                            self.logger.error(traceback.format_exc())
            
            results.extend(batch_results)
        
        results_df = pd.DataFrame(results)
        
        # 保存合并的隐藏状态
        if self.save_hidden_states:
            self.logger.info("正在保存合并的隐藏状态...")
            self.logger.info(f"收集到的隐藏状态方法: {list(all_hidden_states.keys())}")
            
            # 检查是否有收集到的隐藏状态
            if not all_hidden_states:
                self.logger.warning("没有收集到任何隐藏状态，无法保存")
                return results_df
                
            # 检查每个方法收集到的张量数量
            for method, tensors in all_hidden_states.items():
                self.logger.info(f"方法 {method} 收集到 {len(tensors)} 个张量")
                if not tensors:
                    self.logger.warning(f"方法 {method} 没有收集到任何张量")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"hidden_states_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            saved_files = []  # 记录成功保存的文件
            
            for method, tensors in all_hidden_states.items():
                if tensors:
                    try:
                        # 检查所有张量的形状
                        shapes = [t.shape for t in tensors]
                        #self.logger.info(f"方法 {method} 的张量形状: {shapes}")
                        
                        # 检查是否所有张量形状一致
                        if len(set(str(s) for s in shapes)) == 1:
                            # 形状一致，可以直接堆叠
                            stacked_tensor = torch.stack(tensors)
                            # 确保目标词和位置信息是字符串，并替换冒号为下划线
                            safe_target_word = str(self.target_word).replace(':', '_') if self.target_word else "unknown"
                            safe_target_pos = str(self.target_pos).replace(':', '_') if self.target_pos else "unknown"
                            file_path = os.path.join(output_dir, f"hidden_states_{safe_target_word}_{safe_target_pos}_{method}.pt")
                            
                            # 保存前检查张量是否有效
                            if torch.isnan(stacked_tensor).any():
                                #self.logger.warning(f"方法 {method} 的张量包含NaN值，跳过保存")
                                continue
                                
                            if torch.isinf(stacked_tensor).any():
                                #self.logger.warning(f"方法 {method} 的张量包含Inf值，跳过保存")
                                continue
                            
                            torch.save(stacked_tensor, file_path)
                            saved_files.append(file_path)
                            self.logger.info(f"已保存 {len(tensors)} 个隐藏状态到: {file_path}")
                            #self.logger.info(f"保存的张量形状: {stacked_tensor.shape}")
                        else:
                            # 形状不一致，分别保存每个张量
                            #self.logger.warning(f"方法 {method} 的张量形状不一致，无法堆叠。将分别保存每个张量。")
                            method_dir = os.path.join(output_dir, method)
                            os.makedirs(method_dir, exist_ok=True)
                            
                            for i, tensor in enumerate(tensors):
                                # 检查张量是否有效
                                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                    #self.logger.warning(f"张量 {i} 包含NaN或Inf值，跳过保存")
                                    continue
                                    
                                safe_target_word = str(self.target_word).replace(':', '_') if self.target_word else "unknown"
                                safe_target_pos = str(self.target_pos).replace(':', '_') if self.target_pos else "unknown"
                                file_path = os.path.join(method_dir, f"hidden_states_{safe_target_word}_{safe_target_pos}_{i}.pt")
                                torch.save(tensor, file_path)
                                saved_files.append(file_path)
                                
                            self.logger.info(f"已分别保存 {len(tensors)} 个隐藏状态到目录: {method_dir}")
                    except Exception as e:
                        self.logger.error(f"保存 {method} 的隐藏状态时出错: {e}")
                        self.logger.error(traceback.format_exc())
            
            if saved_files:
                self.logger.info(f"成功保存了 {len(saved_files)} 个隐藏状态文件")
                self.logger.info(f"隐藏状态已保存到目录: {output_dir}")
            else:
                self.logger.warning("没有成功保存任何隐藏状态文件")
        
        return results_df
    
    def _process_batch(self, batch_prompts, batch_df):
        batch_results = []
        
        for prompt, (_, row) in zip(batch_prompts, batch_df.iterrows()):
            # 将prompt列表转换为字符串
            if isinstance(prompt, list):
                # 如果是消息列表格式（如ChatML格式），转换为字符串
                prompt_str = ""
                for message in prompt:
                    role = message.get('role', '')
                    content = message.get('content', '')
                    prompt_str += f"{role}: {content}\n"
            else:
                # 如果已经是字符串，直接使用
                prompt_str = prompt
            
            # 处理输入
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
            input_length = inputs['input_ids'].shape[1]  # 记录原始输入长度
            
            # 根据提取方法决定是否需要生成
            need_generation = self.extraction_method in ["output_mean", "output_eos", "all"]
            
            try:
                if need_generation:
                    # 进行生成并保存生成信息
                    with torch.no_grad():
                        generation_outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            num_beams=self.num_beams,
                            temperature=self.temperature,
                            output_hidden_states=True,
                            return_dict_in_generate=True
                        )
                    
                    # 保存生成的完整序列和原始输入长度
                    self.generation_outputs = generation_outputs
                    self.input_length = input_length
                    
                    # 提取生成部分的隐藏状态
                    output_hidden_states = self._extract_hidden_states(
                        generation_outputs.hidden_states,
                        generation_outputs.sequences,
                        method=self.extraction_method
                    )
                    
                    # 生成的文本
                    generated_text = self.tokenizer.decode(
                        generation_outputs.sequences[0][input_length:], 
                        skip_special_tokens=True
                    )
                    
                    # 检查是否需要重新生成（如果提取方法返回need_generation标志）
                    if isinstance(output_hidden_states, dict) and output_hidden_states.get("need_generation", False):
                        #self.logger.info("提取方法需要生成过程，但检测到问题，尝试重新生成")
                        # 已经生成过了，直接重新提取
                        output_hidden_states = self._extract_hidden_states(
                            generation_outputs.hidden_states,
                            generation_outputs.sequences,
                            method=self.extraction_method
                        )
                else:
                    # 如果不需要生成，直接提取输入的隐藏状态
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True)
                    
                    input_hidden_states = self._extract_hidden_states(
                        outputs.hidden_states,
                        inputs['input_ids'],
                        method=self.extraction_method
                    )
                    
                    generated_text = "[No generation needed]"
                    output_hidden_states = input_hidden_states
                
                # 构建结果
                result = {
                    'sentence': row['sentence'],
                    'generated_text': generated_text,
                }
                
                # 添加隐藏状态到结果中
                if output_hidden_states and isinstance(output_hidden_states, dict) and 'hidden_states' in output_hidden_states:
                    result[f'hidden_states_{self.extraction_method}'] = output_hidden_states['hidden_states']
                    
                    # 添加元数据
                    if 'metadata' in output_hidden_states:
                        result[f'metadata_{self.extraction_method}'] = output_hidden_states['metadata']
                    
                    # 添加位置信息
                    if 'positions' in output_hidden_states:
                        result[f'positions_{self.extraction_method}'] = output_hidden_states['positions']
                
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"处理批次时出错: {str(e)}")
                self.logger.error(traceback.format_exc())
                # 添加错误信息到结果
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
        处理input_last_token方法的隐藏状态提取
        
        Args:
            hidden_states: 模型的hidden states
            input_ids: 输入的token ids
            layer_indices: 要提取的层索引列表
            
        Returns:
            包含处理后隐藏状态和元数据的字典，或None（处理失败时）
        """
        try:
            # 验证输入
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states为空")
                return None
                
            if input_ids.numel() == 0:
                #self.logger.error("input_ids为空")
                return None
                
            # 处理layer_indices为None的情况
            if layer_indices is None:
                if self.use_all_layers:
                    layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # 默认使用第一层
                
            states = []
            valid_layers = []
            # Initialize metadata only for extraction_positions
            metadata = {}

            # 获取最后一个非EOS token的位置
            last_token_pos = (input_ids != self.tokenizer.pad_token_id).sum(dim=1) - 1
            
            # 确保位置索引有效
            last_token_pos = torch.clamp(last_token_pos, min=0, max=input_ids.size(1)-1)
            
            # 如果有EOS token，则取其前一个位置
            eos_mask = (input_ids[torch.arange(input_ids.size(0)), last_token_pos] == self.tokenizer.eos_token_id)
            if eos_mask.any():
                #self.logger.info(f"发现{eos_mask.sum().item()}个序列的最后一个token是EOS，将取其前一个位置")
                last_token_pos[eos_mask] = torch.clamp(last_token_pos[eos_mask] - 1, min=0)

            # 记录提取位置信息
            metadata['extraction_positions'] = last_token_pos.tolist()
            #self.logger.info(f"输入序列长度: {input_ids.shape[1]}")
            #self.logger.info(f"实际提取位置: {last_token_pos.tolist()}")
            #self.logger.info(f"隐藏状态层数: {len(hidden_states)}")

            # 确定要处理的层
            valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
            if not valid_layer_indices:
                #self.logger.error(f"没有有效的层索引。请求的索引: {layer_indices}, 可用层数: {len(hidden_states)}")
                return None
                
            #self.logger.info(f"将处理以下层: {valid_layer_indices}")

            for layer_idx in valid_layer_indices:
                try:
                    # 获取当前层的hidden states
                    layer_states = hidden_states[layer_idx]
                    
                    # 处理不同类型的隐藏状态
                    if isinstance(layer_states, tuple):
                        #self.logger.info(f"层{layer_idx}的hidden states是tuple类型，尝试转换")
                        layer_states = torch.stack(list(layer_states))
                    elif not isinstance(layer_states, torch.Tensor):
                        #self.logger.error(f"层{layer_idx}的hidden states类型不支持: {type(layer_states)}")
                        continue
                    
                    # 检查形状
                    #self.logger.info(f"层{layer_idx}的hidden states形状: {layer_states.shape}")
                    
                    if layer_states.dim() < 3:
                        #self.logger.error(f"层{layer_idx}的hidden states维度不足: {layer_states.dim()}")
                        continue
                        
                    if layer_states.size(0) != input_ids.size(0):
                        #self.logger.error(f"层{layer_idx}的batch大小不匹配: {layer_states.size(0)} vs {input_ids.size(0)}")
                        continue

                    # 提取每个序列中最后一个token的hidden states
                    batch_states = []
                    for batch_idx, pos_idx in enumerate(last_token_pos):
                        # 防止位置索引超出范围
                        safe_pos = min(pos_idx.item(), layer_states.size(1)-1)
                        token_state = layer_states[batch_idx, safe_pos]
                        
                        # 检查提取的张量
                        if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                            #self.logger.warning(f"批次{batch_idx}位置{safe_pos}的张量包含NaN或Inf值，将替换为零")
                            token_state = torch.zeros_like(token_state)
                            
                        batch_states.append(token_state)
                    
                    # 堆叠batch维度的结果
                    batch_states = torch.stack(batch_states)
                    states.append(batch_states)
                    valid_layers.append(layer_idx)

                    # 记录统计信息
                    # metadata['last_token_stats'][f'layer_{layer_idx}'] = {
                    #     'mean': float(batch_states.mean().item()),
                    #     'std': float(batch_states.std().item()),
                    #     'min': float(batch_states.min().item()),
                    #     'max': float(batch_states.max().item())
                    # }

                except Exception as e:
                    #self.logger.error(f"处理层{layer_idx}时出错: {str(e)}")
                    #self.logger.error(traceback.format_exc())
                    # metadata['processing_info'].append(f"层{layer_idx}处理失败: {str(e)}")
                    continue

            if not states:
                #self.logger.error("没有收集到任何有效状态")
                return None

            try:
                # 堆叠所有层的结果
                stacked_states = torch.stack(states)  # [num_layers, batch_size, hidden_size]
                
                # 检查堆叠后的张量
                if torch.isnan(stacked_states).any():
                    #self.logger.warning("堆叠后的张量包含NaN值，将替换为零")
                    stacked_states = torch.nan_to_num(stacked_states, nan=0.0)
                if torch.isinf(stacked_states).any():
                    #self.logger.warning("堆叠后的张量包含Inf值，将替换为零")
                    stacked_states = torch.nan_to_num(stacked_states, posinf=0.0, neginf=0.0)
                    
                # 构建结果
                result = {
                    'hidden_states': stacked_states,
                    'valid_layers': valid_layers,
                    # REMOVED: 'metadata': metadata,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids],
                    'positions': last_token_pos.cpu().numpy()
                }

                #self.logger.info(f"最终hidden_states形状: {result['hidden_states'].shape}")
                #self.logger.info(f"处理了{len(valid_layers)}/{len(valid_layer_indices)}个层")

                return result
                
            except Exception as e:
                #self.logger.error(f"堆叠层状态时出错: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"处理input_last_token时出错: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _extract_hidden_states(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        method: str = "input_last_token"
    ) -> Optional[Dict[str, Any]]:
        """提取隐藏状态的增强版本"""
        # 检查hidden_states的结构
        #self.logger.info(f"提取方法: {method}")
        #self.logger.info(f"隐藏状态类型: {type(hidden_states)}")
        #self.logger.info(f"隐藏状态长度: {len(hidden_states)}")
        
        # 检查第一个元素以进一步了解结构
        if len(hidden_states) > 0:
            #self.logger.info(f"第一个元素的类型: {type(hidden_states[0])}, 形状: {hidden_states[0].shape if isinstance(hidden_states[0], torch.Tensor) else None}")
            pass
        
        # 改进的生成模型隐藏状态检测逻辑
        is_generation_output = False
        input_length = getattr(self, 'input_length', None)
        
        # 检查是否为生成模型的输出结构
        if isinstance(hidden_states, list) and len(hidden_states) > 0:
            first_element = hidden_states[0]
            if isinstance(first_element, (tuple, list)) and len(first_element) > 0:
                is_generation_output = True
                #self.logger.info("检测到生成模型的隐藏状态结构 (列表中包含元组或列表)")
        elif isinstance(hidden_states, tuple) and method in ["output_mean", "output_eos"]:
            # 如果是元组，且方法是 output_mean 或 output_eos，可能需要特殊处理
            #self.logger.warning("hidden_states 是一个元组，可能是生成模型的隐藏状态，尝试处理")
            # 检查元组中的元素是否为张量，并且数量与层数匹配
            if len(hidden_states) == self.model.config.num_hidden_layers:
                # 假设这是一个普通隐藏状态元组，不是生成步骤的列表
                #self.logger.info("hidden_states 是一个层的隐藏状态元组，不是生成步骤列表")
                is_generation_output = False
            else:
                # 可能是生成步骤的元组，尝试转换为列表
                #self.logger.info("尝试将元组转换为生成步骤列表")
                hidden_states = list(hidden_states)
                is_generation_output = True
        
        # 检查是否有生成序列和输入长度信息
        has_generation_info = hasattr(self, 'generation_outputs') and hasattr(self, 'input_length')
        
        if method in ["output_mean", "output_eos"] and not is_generation_output and not has_generation_info:
            #self.logger.warning(f"方法 {method} 需要生成模型的隐藏状态，但未检测到正确的结构")
            pass
        
        # 根据提取方法调用相应的处理函数
        try:
            result = None
            if method == "input_last_token":
                result = self._process_input_last_token(hidden_states, input_ids)
            elif method == "eos_token":
                result = self._process_eos_token(hidden_states, input_ids)
            elif method == "input_mean":
                result = self._process_input_mean(hidden_states, input_ids, layer_indices=self.layer_indices)
            elif method == "output_mean":
                result = self._process_output_mean(hidden_states, input_ids, layer_indices=self.layer_indices, is_generation_output=is_generation_output)
            elif method == "output_eos":
                result = self._process_output_eos(hidden_states, input_ids, layer_indices=self.layer_indices, is_generation_output=is_generation_output)
            else:
                #self.logger.error(f"不支持的提取方法: {method}")
                return None
            
            if result is None:
                #self.logger.error(f"方法 {method} 返回了None")
                return None
            
            return result
        except Exception as e:
            #self.logger.error(f"提取隐藏状态时出错: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_eos_token(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None
    ) -> Optional[Dict[str, Any]]:
        """处理eos_token方法的隐藏状态提取
        
        Args:
            hidden_states: 模型的hidden states
            input_ids: 输入的token ids
            layer_indices: 要提取的层索引列表
            
        Returns:
            包含处理后隐藏状态和元数据的字典，或None（处理失败时）
        """
        try:
            # 验证输入
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states为空")
                return None
                
            if input_ids.numel() == 0:
                #self.logger.error("input_ids为空")
                return None
            
            # 处理layer_indices为None的情况
            if layer_indices is None:
                if self.use_all_layers:
                    layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # 默认使用第一层
                
            states = []
            valid_layers = []
            # REMOVED: metadata initialization with stats
            # metadata = {
            #     'eos_stats': {},
            #     'processing_info': []
            # }
            
            # 确定要处理的层
            valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
            if not valid_layer_indices:
                #self.logger.error(f"没有有效的层索引。请求的索引: {layer_indices}, 可用层数: {len(hidden_states)}")
                return None
                
            #self.logger.info(f"将处理以下层: {valid_layer_indices}")
            #self.logger.info(f"输入序列形状: {input_ids.shape}")

            # 找到EOS token的位置
            eos_positions = (input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)
            if eos_positions[0].numel() > 0:
                #self.logger.info(f"找到EOS token位置: 批次索引={eos_positions[0].tolist()}, 序列位置={eos_positions[1].tolist()}")
                # 对每个序列分别处理
                eos_pos_by_batch = {}
                for i in range(len(eos_positions[0])):
                    batch_idx = eos_positions[0][i].item()
                    pos = eos_positions[1][i].item()
                    if batch_idx not in eos_pos_by_batch:
                        eos_pos_by_batch[batch_idx] = pos
            else:
                #self.logger.warning("未找到任何EOS token，将使用序列末尾位置")
                eos_pos_by_batch = {}

            for layer_idx in valid_layer_indices:
                try:
                    layer_states = hidden_states[layer_idx]
                    
                    # 处理不同类型的隐藏状态
                    if isinstance(layer_states, tuple):
                        #self.logger.info(f"层{layer_idx}的hidden states是tuple类型，尝试转换")
                        layer_states = torch.stack(list(layer_states))
                    elif not isinstance(layer_states, torch.Tensor):
                        #self.logger.error(f"层{layer_idx}的hidden states类型不支持: {type(layer_states)}")
                        continue
                        
                    # 检查形状
                    #self.logger.info(f"层{layer_idx}的hidden states形状: {layer_states.shape}")
                    
                    if layer_states.dim() < 3:
                        #self.logger.error(f"层{layer_idx}的hidden states维度不足: {layer_states.dim()}")
                        continue
                        
                    if layer_states.size(0) != input_ids.size(0):
                        #self.logger.error(f"层{layer_idx}的batch大小不匹配: {layer_states.size(0)} vs {input_ids.size(0)}")
                        continue
                    
                    # 提取每个序列中EOS token的hidden states
                    batch_states = []
                    for batch_idx in range(input_ids.size(0)):
                        if batch_idx in eos_pos_by_batch:
                            # 使用找到的EOS位置
                            pos = eos_pos_by_batch[batch_idx]
                        else:
                            # 如果没有EOS token，使用最后一个非padding token
                            pos = (input_ids[batch_idx] != self.tokenizer.pad_token_id).sum().item() - 1
                            pos = max(0, pos)  # 确保位置不为负
                        
                        # 确保位置在有效范围内
                        safe_pos = min(pos, layer_states.size(1) - 1)
                        token_state = layer_states[batch_idx, safe_pos]
                        
                        # 检查提取的张量
                        if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                            #self.logger.warning(f"批次{batch_idx}位置{safe_pos}的张量包含NaN或Inf值，将替换为零")
                            token_state = torch.zeros_like(token_state)
                            
                        batch_states.append(token_state)
                    
                    # 堆叠batch维度的结果
                    batch_states = torch.stack(batch_states)
                    states.append(batch_states)
                    valid_layers.append(layer_idx)
                    
                    # 记录统计信息
                    # metadata['eos_stats'][f'layer_{layer_idx}'] = {
                    #     'mean': float(batch_states.mean().item()),
                    #     'std': float(batch_states.std().item()),
                    #     'min': float(batch_states.min().item()),
                    #     'max': float(batch_states.max().item())
                    # }

                except Exception as e:
                    #self.logger.error(f"处理层{layer_idx}时出错: {str(e)}")
                    #self.logger.error(traceback.format_exc())
                    # REMOVED: processing_info append
                    # metadata['processing_info'].append(f"层{layer_idx}处理失败: {str(e)}")
                    continue

            if not states:
                #self.logger.error("没有收集到任何有效状态")
                return None
            
            # 构建结果
            result = {
                'hidden_states': torch.stack(states),  # [num_layers, batch_size, hidden_size]
                'valid_layers': valid_layers,
                # REMOVED: 'metadata': metadata,
                'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
            }
            
            #self.logger.info(f"最终hidden_states形状: {result['hidden_states'].shape}")
            #self.logger.info(f"处理了{len(valid_layers)}/{len(valid_layer_indices)}个层")
            
            return result
        except Exception as e:
            #self.logger.error(f"处理eos_token时出错: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_input_mean(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None
    ) -> Optional[Dict[str, Any]]:
        """处理input_mean方法的隐藏状态提取"""
        try:
            # 验证输入
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states为空")
                return None
            
            if input_ids.numel() == 0:
                #self.logger.error("input_ids为空")
                return None
            
            # 处理layer_indices为None的情况
            if layer_indices is None:
                if self.use_all_layers:
                    layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # 默认使用第一层
            
            states = []
            all_mean_states = []  # 添加这个变量
            valid_layers = []
             # REMOVED: metadata initialization with stats
            # metadata = {
            #     'mean_stats': {},
            #     'processing_info': []
            # }
             
            # 创建attention mask来忽略padding tokens
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
            #self.logger.info(f"输入序列形状: {input_ids.shape}")
            #self.logger.info(f"注意力掩码形状: {attention_mask.shape}")
            
            # 确定要处理的层
            valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
            if not valid_layer_indices:
                #self.logger.error(f"没有有效的层索引。请求的索引: {layer_indices}, 可用层数: {len(hidden_states)}")
                return None
            
            #self.logger.info(f"将处理以下层: {valid_layer_indices}")

            for layer_idx in valid_layer_indices:
                try:
                    layer_states = hidden_states[layer_idx]
                    
                    # 处理不同类型的隐藏状态
                    if isinstance(layer_states, tuple):
                        #self.logger.info(f"层{layer_idx}的hidden states是tuple类型，尝试转换")
                        layer_states = torch.stack(list(layer_states))
                    elif not isinstance(layer_states, torch.Tensor):
                        #self.logger.error(f"层{layer_idx}的hidden states类型不支持: {type(layer_states)}")
                        continue
                    
                    # 检查形状
                    #self.logger.info(f"层{layer_idx}的hidden states形状: {layer_states.shape}")
                    
                    if layer_states.dim() < 3:
                        #self.logger.error(f"层{layer_idx}的hidden states维度不足: {layer_states.dim()}")
                        continue
                    
                    if layer_states.size(0) != input_ids.size(0):
                        #self.logger.error(f"层{layer_idx}的batch大小不匹配: {layer_states.size(0)} vs {input_ids.size(0)}")
                        continue
                    
                    # 计算平均值（忽略padding tokens）
                    try:
                        # 扩展注意力掩码以匹配隐藏状态的维度
                        expanded_mask = attention_mask.unsqueeze(-1)
                        
                        # 应用掩码
                        masked_states = layer_states * expanded_mask
                        
                        # 计算每个序列的平均值
                        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                        mean_states = masked_states.sum(dim=1) / sum_mask
                        
                        # 检查是否有NaN或Inf
                        if torch.isnan(mean_states).any() or torch.isinf(mean_states).any():
                            self.logger.warning(f"层{layer_idx}的平均隐藏状态包含NaN或Inf，将替换为零")
                            mean_states = torch.zeros_like(mean_states)
                        
                        
                        all_mean_states.append(mean_states)
                        valid_layers.append(layer_idx)
                        
                        # 修改日志记录平均后的隐藏状态形状
                        #self.logger.info(f"层{layer_idx}的平均hidden states形状: {mean_states.shape}")
                    except Exception as e:
                        #self.logger.error(f"计算层{layer_idx}的平均状态时出错: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        continue
                    
                    states.append(mean_states)
                    
                    # 记录统计信息
                    # metadata['mean_stats'][f'layer_{layer_idx}'] = {
                    #     'mean': float(mean_states.mean().item()),
                    #     'std': float(mean_states.std().item()),
                    #     'min': float(mean_states.min().item()),
                    #     'max': float(mean_states.max().item())
                    # }

                except Exception as e:
                    #self.logger.error(f"处理层{layer_idx}时出错: {str(e)}")
                    #self.logger.error(traceback.format_exc())
                    # REMOVED: processing_info append
                    # metadata['processing_info'].append(f"层{layer_idx}处理失败: {str(e)}")
                    continue

            if not states:
                #self.logger.error("没有收集到任何有效状态")
                return None
            
            # 堆叠所有层
            try:
                stacked_all_layers = torch.stack(all_mean_states)
                
                # 构建结果
                result = {
                    'hidden_states': stacked_all_layers,  # [num_layers, batch_size, hidden_size]
                    'valid_layers': valid_layers,
                    # REMOVED: 'metadata': metadata,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
                }
                
                #self.logger.info(f"最终平均hidden_states形状: {result['hidden_states'].shape}")
                #self.logger.info(f"处理了{len(valid_layers)}/{len(valid_layer_indices)}个层")
                
                return result
            except Exception as e:
                #self.logger.error(f"构建结果时出错: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"处理input_mean时出错: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_output_mean(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None,
        is_generation_output: bool = False
    ) -> Optional[Dict[str, Any]]:
        """处理output_mean方法的隐藏状态提取"""
        try:
            # 验证输入
            if not hidden_states:
                #self.logger.error("hidden_states为空")
                return None
                
            if input_ids.numel() == 0:
                #self.logger.error("input_ids为空")
                return None
        
            # 获取输入序列长度
            input_length = getattr(self, 'input_length', None)
            if input_length is None or input_length <= 0:
                #self.logger.warning("无法获取有效的输入长度，将使用启发式方法")
                input_length = input_ids.shape[1] // 2  # 假设生成部分占总长度的一半
        
            # 获取总序列长度
            total_length = input_ids.shape[1]  # 直接使用input_ids的长度作为总长度
        
            #self.logger.info(f"输入序列长度: {input_length}, 总序列长度: {total_length}, 生成部分长度: {total_length - input_length}")
        
            # 检查是否有生成部分
            if total_length <= input_length:
                #self.logger.error(f"总长度({total_length})不大于输入长度({input_length})，无法提取生成部分")
                return None
        
            # 检查hidden_states的结构
            #self.logger.info(f"hidden_states类型: {type(hidden_states)}")
            #self.logger.info(f"hidden_states长度: {len(hidden_states)}")
            #self.logger.info(f"是否为生成模型输出: {is_generation_output}")
        
            all_mean_states = []
            valid_layers = []
            
            # 处理layer_indices为None的情况
            if layer_indices is None:
                if self.use_all_layers:
                    # 如果使用所有层，根据隐藏状态结构确定层数
                    if is_generation_output and len(hidden_states) > 0 and isinstance(hidden_states[0], (tuple, list)):
                        layer_indices = list(range(len(hidden_states[0])))
                    else:
                        layer_indices = list(range(len(hidden_states)))
                else:
                    # 如果不使用所有层但未指定层索引，使用默认值
                    layer_indices = [0]  # 默认使用第一层
                    
            #self.logger.info(f"使用的层索引: {layer_indices}")
            
            if is_generation_output:
                # 处理生成模型的隐藏状态结构
                #self.logger.info("处理生成模型的隐藏状态结构")
                
                # 确定有效的层索引
                if len(hidden_states) > 0 and isinstance(hidden_states[0], (tuple, list)):
                    num_layers = len(hidden_states[0])
                    valid_layer_indices = [idx for idx in layer_indices if idx < num_layers]
                else:
                    #self.logger.error("生成模型的隐藏状态结构不符合预期")
                    return None
                
                if not valid_layer_indices:
                    #self.logger.error(f"没有有效的网络层索引。请求的索引: {layer_indices}, 可用网络层数: {num_layers}")
                    return None
                
                #self.logger.info(f"将处理以下网络层: {valid_layer_indices}")
                
                # 对每一层处理所有生成步骤
                for layer_idx in valid_layer_indices:
                    try:
                        # 收集该层所有生成步骤的隐藏状态
                        layer_states = []
                        
                        for step_idx, step_hidden_states in enumerate(hidden_states):
                            if layer_idx < len(step_hidden_states):
                                current_layer_state = step_hidden_states[layer_idx]
                                
                                # 处理不同类型的隐藏状态
                                if isinstance(current_layer_state, tuple):
                                    #self.logger.info(f"步骤{step_idx}网络层{layer_idx}的隐藏状态是tuple类型，尝试获取第一个元素")
                                    if len(current_layer_state) > 0:
                                        current_layer_state = current_layer_state[0]
                                    else:
                                        continue
                                
                                if not isinstance(current_layer_state, torch.Tensor):
                                    #self.logger.warning(f"步骤{step_idx}网络层{layer_idx}的隐藏状态类型不支持: {type(current_layer_state)}")
                                    continue
                                
                                # 检查形状
                                if len(current_layer_state.shape) != 3:
                                    #self.logger.warning(f"步骤{step_idx}网络层{layer_idx}的隐藏状态维度不正确: {len(current_layer_state.shape)}，期望3维")
                                    continue
                                
                                # 对于生成模型，每个步骤只有一个token，我们直接收集所有步骤的隐藏状态
                                layer_states.append(current_layer_state)
                        
                        if not layer_states:
                            #self.logger.warning(f"网络层{layer_idx}没有收集到有效的隐藏状态")
                            continue
                        
                        # 将所有步骤的隐藏状态拼接在一起
                        concatenated_states = torch.cat(layer_states, dim=1)
                        #self.logger.info(f"网络层{layer_idx}拼接后的隐藏状态形状: {concatenated_states.shape}")
                        
                        # 提取生成部分（如果有输入长度信息）
                        if input_length > 0 and concatenated_states.shape[1] > input_length:
                            generated_part = concatenated_states[:, input_length:, :]
                            #self.logger.info(f"网络层{layer_idx}提取生成部分，形状: {generated_part.shape}")
                        else:
                            # 如果没有明确的输入长度，假设所有步骤都是生成的
                            generated_part = concatenated_states
                            #self.logger.info(f"网络层{layer_idx}使用所有隐藏状态，形状: {generated_part.shape}")
                        
                        # 计算平均值
                        mean_output_hidden_states = torch.mean(generated_part, dim=1)
                        
                        # 检查是否有NaN或Inf
                        if torch.isnan(mean_output_hidden_states).any() or torch.isinf(mean_output_hidden_states).any():
                            #self.logger.warning(f"网络层{layer_idx}的平均隐藏状态包含NaN或Inf，将替换为零")
                            mean_output_hidden_states = torch.zeros_like(mean_output_hidden_states)
                        
                        all_mean_states.append(mean_output_hidden_states)
                        valid_layers.append(layer_idx)
                    except Exception as e:
                        #self.logger.error(f"处理生成模型网络层{layer_idx}时出错: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        continue
            else:
                # 处理标准模型的隐藏状态
                # 确定要处理的网络层索引
                valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
                if not valid_layer_indices:
                    #self.logger.error(f"没有有效的网络层索引。请求的索引: {layer_indices}, 可用网络层数: {len(hidden_states)}")
                    return None
                
                #self.logger.info(f"将处理以下网络层: {valid_layer_indices}")

                # 简化处理逻辑，不区分生成模型和标准模型
                for layer_idx in valid_layer_indices:
                    try:
                        # 获取当前网络层的隐藏状态
                        if layer_idx >= len(hidden_states):
                            continue
                    
                        current_layer_states = hidden_states[layer_idx]
                    
                        # 处理不同类型的隐藏状态
                        if isinstance(current_layer_states, tuple):
                            #self.logger.info(f"网络层{layer_idx}的隐藏状态是tuple类型，尝试获取第一个元素")
                            if len(current_layer_states) > 0:
                                current_layer_states = current_layer_states[0]
                            else:
                                #self.logger.warning(f"网络层{layer_idx}的隐藏状态是空tuple")
                                continue
                    
                        if not isinstance(current_layer_states, torch.Tensor):
                            #self.logger.error(f"网络层{layer_idx}的隐藏状态类型不支持: {type(current_layer_states)}")
                            continue
                    
                        # 检查形状
                        #self.logger.info(f"网络层{layer_idx}的隐藏状态形状: {current_layer_states.shape}")
                    
                        # 检查维度是否符合预期
                        if len(current_layer_states.shape) != 3:
                            #self.logger.error(f"网络层{layer_idx}的隐藏状态维度不正确: {len(current_layer_states.shape)}，期望3维")
                            continue
                    
                        # 检查序列长度
                        seq_length = current_layer_states.shape[1]
                    
                        # 如果序列长度为1，这可能是生成模型的中间状态，不适合提取
                        if seq_length == 1:
                            #self.logger.warning(f"网络层{layer_idx}的序列长度为1，可能是中间状态，跳过")
                            continue
                    
                        # 提取生成部分
                        if seq_length >= total_length:
                            # 如果隐藏状态长度大于等于总长度，使用总长度作为参考
                            generated_part = current_layer_states[:, input_length:total_length, :]
                        elif seq_length > input_length:
                            # 如果隐藏状态长度大于输入长度但小于总长度，使用隐藏状态自身的长度
                            #self.logger.warning(f"网络层{layer_idx}的隐藏状态长度({seq_length})小于总长度({total_length})，使用隐藏状态长度")
                            generated_part = current_layer_states[:, input_length:, :]
                        else:
                            # 如果隐藏状态长度小于等于输入长度，无法提取生成部分
                            #self.logger.warning(f"网络层{layer_idx}的隐藏状态长度({seq_length})不足以提取生成部分(输入长度:{input_length})")
                            continue
                    
                        # 检查生成部分是否为空
                        if generated_part.shape[1] == 0:
                            #self.logger.warning(f"网络层{layer_idx}的生成部分为空")
                            continue
                    
                        #self.logger.info(f"网络层{layer_idx}提取生成部分，形状: {generated_part.shape}")
                    
                        # 计算平均值
                        mean_output_hidden_states = torch.mean(generated_part, dim=1)
                    
                        # 检查是否有NaN或Inf
                        if torch.isnan(mean_output_hidden_states).any() or torch.isinf(mean_output_hidden_states).any():
                            #self.logger.warning(f"网络层{layer_idx}的平均隐藏状态包含NaN或Inf，将替换为零")
                            mean_output_hidden_states = torch.zeros_like(mean_output_hidden_states)
                    
                        all_mean_states.append(mean_output_hidden_states)
                        valid_layers.append(layer_idx)
                    except Exception as e:
                        #self.logger.error(f"处理网络层{layer_idx}时出错: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        continue
        
            if not all_mean_states:
                #self.logger.error("没有收集到有效状态")
                return None
        
            # 堆叠所有网络层
            try:
                stacked_all_layers = torch.stack(all_mean_states)
                
                # 构建结果
                result = {
                    'hidden_states': stacked_all_layers,  # [num_layers, batch_size, hidden_size]
                    'valid_layers': valid_layers,
                    # REMOVED: 'metadata': metadata,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
                }
                
                #self.logger.info(f"最终平均hidden_states形状: {result['hidden_states'].shape}")
                #self.logger.info(f"处理了{len(valid_layers)}/{len(valid_layer_indices)}个网络层")
                
                return result
            except Exception as e:
                #self.logger.error(f"堆叠隐藏状态时出错: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"处理output_mean时出错: {str(e)}")
            #self.logger.error(traceback.format_exc())
            return None

    def _process_output_eos(
        self,
        hidden_states: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        input_ids: torch.Tensor,
        layer_indices: List[int] = None,
        is_generation_output: bool = False
    ) -> Optional[Dict[str, Any]]:
        """处理output_eos方法的隐藏状态提取 - 简化版本，只提取生成部分的最后一个token"""
        try:
            # 验证输入
            if len(hidden_states) == 0:
                #self.logger.error("hidden_states为空")
                return None
    
            if input_ids.numel() == 0:
                #self.logger.error("input_ids为空")
                return None
        
             # REMOVED: metadata initialization with stats
            # metadata = {
            #     'last_token_stats': {},
            #     'processing_info': []
            # }
              
            # 获取输入序列长度
            input_length = getattr(self, 'input_length', None)
            if input_length is None:
                #self.logger.warning("无法确定输入长度，使用启发式方法")
                input_length = max(1, input_ids.shape[1] // 2)  # 假设生成部分占总长度的一半
    
            #self.logger.info(f"输入序列长度: {input_length}, 总序列长度: {input_ids.shape[1]}")
    
            # 处理layer_indices为None的情况
            if layer_indices is None:
                if self.use_all_layers:
                    # 根据隐藏状态结构确定层数
                    if is_generation_output and len(hidden_states) > 0 and isinstance(hidden_states[-1], (tuple, list)):
                        layer_indices = list(range(len(hidden_states[-1])))
                    else:
                        layer_indices = list(range(len(hidden_states)))
                else:
                    layer_indices = [0]  # 默认使用第一层
            
            #self.logger.info(f"使用的层索引: {layer_indices}")
        
            # 为每个序列找到生成部分的最后一个token位置
            last_token_positions = []
            for batch_idx, seq in enumerate(input_ids):
                # 找到每个序列中的最后一个非padding token
                seq_length = (seq != self.tokenizer.pad_token_id).sum().item()
            
                # 如果序列长度大于输入长度，则最后一个token在生成部分
                if seq_length > input_length:
                    # 使用生成部分的最后一个token (seq_length - 1)
                    last_token_positions.append(seq_length - 1)
                else:
                    # 如果序列长度小于等于输入长度，可能没有生成部分，使用输入的最后位置
                    #self.logger.warning(f"序列{batch_idx}可能没有生成部分，使用输入末尾位置")
                    last_token_positions.append(max(0, seq_length - 1))
            
            #self.logger.info(f"生成部分最后token位置: {last_token_positions}")
        
            # 用于收集隐藏状态
            all_last_token_states = []
            valid_layers = []
        
            # 处理隐藏状态，根据是否为生成模型输出采用不同策略
            if is_generation_output:
                # 对于生成模型，使用最后一步的隐藏状态
                #self.logger.info("处理生成模型的隐藏状态")
            
                # 获取最后一步的隐藏状态
                last_step_hidden_states = hidden_states[-1]
            
                # 确定有效层索引
                if isinstance(last_step_hidden_states, (tuple, list)):
                    available_layers = len(last_step_hidden_states)
                else:
                    #self.logger.error(f"生成模型的最后一步隐藏状态类型错误: {type(last_step_hidden_states)}")
                    return None
            
                valid_layer_indices = [idx for idx in layer_indices if idx < available_layers]
                if not valid_layer_indices:
                    #self.logger.error(f"没有有效的层索引。请求的索引: {layer_indices}, 可用层数: {available_layers}")
                    return None
            
                #self.logger.info(f"将处理以下层: {valid_layer_indices}")
            
                # 处理每一层
                for layer_idx in valid_layer_indices:
                    try:
                        # 获取当前层的隐藏状态
                        if isinstance(last_step_hidden_states, (tuple, list)):
                            layer_states = last_step_hidden_states[layer_idx]
                        else:
                            continue
                    
                        # 处理tuple类型的隐藏状态
                        if isinstance(layer_states, tuple) and len(layer_states) > 0:
                            layer_states = layer_states[0]
                        
                        if not isinstance(layer_states, torch.Tensor):
                            #self.logger.error(f"层{layer_idx}的隐藏状态类型不支持: {type(layer_states)}")
                            continue
                        
                        #self.logger.info(f"层{layer_idx}的hidden states形状: {layer_states.shape}")
                    
                        # 提取最后token的隐藏状态
                        batch_states = []
                        for batch_idx, pos in enumerate(last_token_positions):
                            if batch_idx >= layer_states.shape[0]:
                                continue
                            
                            # 确保位置在有效范围内
                            safe_pos = min(pos, layer_states.shape[1] - 1)
                            token_state = layer_states[batch_idx, safe_pos]
                            
                            # 检查并处理NaN或Inf
                            if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                                #self.logger.warning(f"层{layer_idx}批次{batch_idx}位置{safe_pos}的张量包含NaN或Inf，将替换为零")
                                token_state = torch.zeros_like(token_state)
                                
                            batch_states.append(token_state)
                        
                        if batch_states:
                            # 堆叠所有批次的结果
                            stacked_states = torch.stack(batch_states)
                            all_last_token_states.append(stacked_states)
                            valid_layers.append(layer_idx)
                            
                            # 记录统计信息
                            # metadata['last_token_stats'][f'layer_{layer_idx}'] = {
                            #     'mean': float(stacked_states.mean().item()),
                            #     'std': float(stacked_states.std().item()),
                            #     'min': float(stacked_states.min().item()),
                            #     'max': float(stacked_states.max().item())
                            # }
                    except Exception as e:
                        #self.logger.error(f"处理层{layer_idx}时出错: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        # REMOVED: processing_info append
                        # metadata['processing_info'].append(f"层{layer_idx}处理失败: {str(e)}")
                        continue
            else:
                # 处理标准模型的隐藏状态
                #self.logger.info("处理标准模型的隐藏状态")
            
                valid_layer_indices = [idx for idx in layer_indices if idx < len(hidden_states)]
                if not valid_layer_indices:
                    #self.logger.error(f"没有有效的层索引。请求的索引: {layer_indices}, 可用层数: {len(hidden_states)}")
                    return None
            
                #self.logger.info(f"将处理以下层: {valid_layer_indices}")
            
                # 处理每一层
                for layer_idx in valid_layer_indices:
                    try:
                        layer_states = hidden_states[layer_idx]
                    
                        # 处理tuple类型的隐藏状态
                        if isinstance(layer_states, tuple) and len(layer_states) > 0:
                            layer_states = layer_states[0]
                        
                        if not isinstance(layer_states, torch.Tensor):
                            #self.logger.error(f"层{layer_idx}的隐藏状态类型不支持: {type(layer_states)}")
                            continue
                        
                        #self.logger.info(f"层{layer_idx}的hidden states形状: {layer_states.shape}")
                    
                        # 提取最后token的隐藏状态
                        batch_states = []
                        for batch_idx, pos in enumerate(last_token_positions):
                            if batch_idx >= layer_states.shape[0]:
                                continue
                            
                            # 确保位置在有效范围内
                            safe_pos = min(pos, layer_states.shape[1] - 1)
                            token_state = layer_states[batch_idx, safe_pos]
                            
                            # 检查并处理NaN或Inf
                            if torch.isnan(token_state).any() or torch.isinf(token_state).any():
                                #self.logger.warning(f"层{layer_idx}批次{batch_idx}位置{safe_pos}的张量包含NaN或Inf，将替换为零")
                                token_state = torch.zeros_like(token_state)
                                
                            batch_states.append(token_state)
                        
                        if batch_states:
                            # 堆叠所有批次的结果
                            stacked_states = torch.stack(batch_states)
                            all_last_token_states.append(stacked_states)
                            valid_layers.append(layer_idx)
                            
                            # 记录统计信息
                            # metadata['last_token_stats'][f'layer_{layer_idx}'] = {
                            #     'mean': float(stacked_states.mean().item()),
                            #     'std': float(stacked_states.std().item()),
                            #     'min': float(stacked_states.min().item()),
                            #     'max': float(stacked_states.max().item())
                            # }
                    except Exception as e:
                        #self.logger.error(f"处理层{layer_idx}时出错: {str(e)}")
                        #self.logger.error(traceback.format_exc())
                        # REMOVED: processing_info append
                        # metadata['processing_info'].append(f"层{layer_idx}处理失败: {str(e)}")
                        continue
        
            # 检查是否收集到有效状态
            if not all_last_token_states:
                #self.logger.error("没有收集到任何有效的最后token隐藏状态")
                return None
        
            # 堆叠所有层的结果并返回
            try:
                stacked_all_layers = torch.stack(all_last_token_states)
                
                # 构建结果
                result = {
                    'hidden_states': stacked_all_layers,  # [num_layers, batch_size, hidden_size]
                    'valid_layers': valid_layers,
                    # REMOVED: 'metadata': metadata,
                    'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids],
                    'positions': last_token_positions
                }
                
                #self.logger.info(f"最终隐藏状态形状: {result['hidden_states'].shape}")
                #self.logger.info(f"处理了{len(valid_layers)}/{len(valid_layer_indices)}个层")
                
                return result
            except Exception as e:
                #self.logger.error(f"构建结果时出错: {str(e)}")
                #self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            #self.logger.error(f"处理output_eos时出错: {str(e)}")
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
        保存结果到文件
        
        Args:
            results: 包含结果的字典
            experiment_name: 实验名称
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            self.logger.info(f"保存结果到: {output_dir}")
            
            # 如果结果是DataFrame，处理并保存
            if isinstance(results, pd.DataFrame):
                # 创建一个副本，避免修改原始数据
                df = results.copy()
                
                # 检查是否有tensor列需要单独保存
                tensor_columns = []
                for col in df.columns:
                    if isinstance(df[col].iloc[0], torch.Tensor):
                        tensor_columns.append(col)
                
                # 如果有tensor列，单独保存
                if tensor_columns:
                    self.logger.info(f"发现tensor列: {tensor_columns}")
                    
                    # 为每个tensor列创建目录
                    for col in tensor_columns:
                        tensor_dir = os.path.join(output_dir, col)
                        os.makedirs(tensor_dir, exist_ok=True)
                        
                        # 保存每个tensor
                        for idx, row in df.iterrows():
                            tensor = row[col]
                            tensor_path = os.path.join(tensor_dir, f"{idx}.pt")
                            torch.save(tensor, tensor_path)
                            self.logger.debug(f"保存tensor到: {tensor_path}")
                        
                        # 从DataFrame中移除tensor列
                        df = df.drop(columns=[col])
                
                # 保存处理后的DataFrame
                csv_path = os.path.join(output_dir, f"{experiment_name}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"保存CSV到: {csv_path}")
                
                return output_dir
            
            # 如果结果是字典，保存为JSON
            elif isinstance(results, dict):
                # 处理字典中的tensor
                processed_results = {}
                
                for key, value in results.items():
                    if isinstance(value, torch.Tensor):
                        # 保存tensor到文件
                        tensor_path = os.path.join(output_dir, f"{key}.pt")
                        torch.save(value, tensor_path)
                        self.logger.info(f"保存tensor到: {tensor_path}")
                        
                        # 在结果字典中记录路径
                        processed_results[key] = f"tensor_saved_at:{tensor_path}"
                    else:
                        processed_results[key] = value
                
                # 保存处理后的字典为JSON
                json_path = os.path.join(output_dir, f"{experiment_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_results, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"保存JSON到: {json_path}")
                
                return output_dir
            
            else:
                error_msg = f"不支持的结果类型: {type(results)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            self.logger.error(f"保存结果时出错: {str(e)}")
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
        将所有隐藏状态合并保存到单个张量文件中
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查张量列
        tensor_columns = []
        for col in results_df.columns:
            if len(results_df) > 0 and isinstance(results_df[col].iloc[0], torch.Tensor):
                tensor_columns.append(col)
        
        self.logger.info(f"找到 {len(tensor_columns)} 个张量列: {tensor_columns}")
        
        # 处理每个张量列
        for col in tensor_columns:
            try:
                # 堆叠列中的所有张量
                tensors = [tensor for tensor in results_df[col] if tensor is not None]
                if not tensors:
                    self.logger.warning(f"列 {col} 中没有找到有效张量")
                    continue
                    
                stacked_tensor = torch.stack(tensors)
                
                # 保存合并的张量
                file_name = f"hidden_states_{col.replace('/', '_')}.pt"
                file_path = os.path.join(output_dir, file_name)
                torch.save(stacked_tensor, file_path)
                self.logger.info(f"已将合并的隐藏状态保存到: {file_path}")
                
            except Exception as e:
                self.logger.error(f"保存 {col} 的合并隐藏状态时出错: {e}")
                self.logger.error(traceback.format_exc())
        
        return output_dir
