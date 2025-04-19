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

# 设置日志配置
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
    """加载单个样本文件，支持.npy和.pt格式，并统一处理形状"""
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.pt'):
            data = torch.load(file_path, map_location='cpu').cpu().numpy()
        else:
            logger.warning(f"不支持的文件格式: {file_path}")
            return None
            
        # 统一处理数据形状：如果第三维大于1，取第一个元素
        if data.ndim == 4 and data.shape[2] > 1:
            logger.info(f"检测到第三维大于1的数据: {data.shape}，取第三维的第一个元素")
            # 取第三维的第一个元素，并保持四维结构
            data = data[:, :, 0:1, :]  # 这样可以保持四维结构
            logger.info(f"处理后的数据形状: {data.shape}")
            
        return data
    except Exception as e:
        logger.error(f"加载文件 {file_path} 时发生错误: {e}")
        return None

def load_method_samples(method_folder: str, file_pattern: str = "*.pt") -> Union[np.ndarray, None]:
    """加载给定方法文件夹中的所有样本文件"""
    file_paths = glob.glob(os.path.join(method_folder, file_pattern))
    
    if not file_paths:
        logger.warning(f"在 {method_folder} 中没有找到匹配 {file_pattern} 的文件")
        return None
    
    # 如果只有一个文件，直接加载该文件
    if len(file_paths) == 1:
        logger.info(f"在 {method_folder} 中找到单个文件，直接加载")
        return load_sample(file_paths[0])
    
    # 使用多进程加载多个文件
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(load_sample, file_paths), 
                          total=len(file_paths),
                          desc=f"加载 {os.path.basename(method_folder)} 中的样本"))
    
    # 过滤掉加载失败的样本并合并
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        logger.warning(f"在 {method_folder} 中没有成功加载任何样本")
        return None
    
    try:
        return np.concatenate(valid_results, axis=0)
    except ValueError as e:
        logger.error(f"合并样本失败: {e}")
        return None

def compute_cosine_similarity_with_layers(v1: np.ndarray, v2: np.ndarray, layer_indices: List[int]) -> float:
    """计算两个样本在指定层上的余弦相似度"""
    # 确保输入是四维张量 [batch=1, layers, 1, hid_dim]
    assert v1.ndim == 4 and v2.ndim == 4, "输入张量必须是四维"
    assert v1.shape[0] == 1 and v2.shape[0] == 1, "批次大小必须为1"
    
    n_layers = v1.shape[1]
    
    # 确保层索引有效
    valid_indices = [l for l in layer_indices if 0 <= l < n_layers]
    if not valid_indices:
        logger.warning(f"所有提供的层索引 {layer_indices} 都超出范围 [0, {n_layers-1}]")
        return 0.0
    
    # 移除第三维(值为1的维度)并选择特定层
    similarities = []
    for l in valid_indices:
        v1_vec = v1[0, l].reshape(1, -1)
        v2_vec = v2[0, l].reshape(1, -1)
        sim = cosine_similarity(v1_vec, v2_vec)[0][0]
        similarities.append(sim)
    
    # 返回所有选定层的平均相似度
    return np.mean(similarities)

def compute_batch_correlation(data_i: np.ndarray, data_j: np.ndarray, 
                             batch_start: int, batch_end: int, 
                             layer_indices: List[int]) -> np.ndarray:
    """计算一批样本的相关性"""
    batch_data_i = data_i[batch_start:batch_end]
    batch_data_j = data_j[batch_start:batch_end]
    
    batch_size = batch_data_i.shape[0]
    batch_similarity = np.zeros(batch_size)
    
    for k in range(batch_size):
        v_i = batch_data_i[k:k+1]  # 保持四维形状 [1, layers, 1, hid_dim]
        v_j = batch_data_j[k:k+1]  # 保持四维形状 [1, layers, 1, hid_dim]
        batch_similarity[k] = compute_cosine_similarity_with_layers(v_i, v_j, layer_indices)
    
    return batch_similarity

def parallel_batch_compute_correlation(method_data: List[np.ndarray], 
                                      batch_size: int, 
                                      layer_indices: List[int]) -> np.ndarray:
    """并行计算多个方法之间的相关性矩阵"""
    num_methods = len(method_data)
    num_samples = min(m.shape[0] for m in method_data)  # 使用所有方法中最小的样本数
    
    # 记录每个方法的形状信息
    for i, data in enumerate(method_data):
        logger.info(f"方法 {i+1} 数据形状: {data.shape}")
    
    logger.info(f"使用 {num_samples} 个样本和层 {layer_indices} 计算相关性")
    
    # 截断所有方法数据至相同的样本数
    method_data = [m[:num_samples] for m in method_data]
    
    # 初始化相关性矩阵并设置对角线为1.0
    correlations = np.zeros((num_methods, num_methods))
    for i in range(num_methods):
        correlations[i, i] = 1.0
    
    # 创建多进程池计算相关性
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i in range(num_methods):
            for j in range(i+1, num_methods):
                data_i = method_data[i]
                data_j = method_data[j]
                
                logger.info(f"计算方法 {i+1} 与方法 {j+1} 之间的相关性")
                
                # 创建批次任务
                results = []
                for batch_start in range(0, num_samples, batch_size):
                    batch_end = min(batch_start + batch_size, num_samples)
                    results.append(
                        pool.apply_async(
                            compute_batch_correlation,
                            (data_i, data_j, batch_start, batch_end, layer_indices)
                        )
                    )
                
                # 收集结果并计算平均相似度
                batch_similarities = [result.get() for result in results]
                all_similarities = np.concatenate(batch_similarities)
                avg_similarity = np.mean(all_similarities)
                
                # 填充相关性矩阵（对称矩阵）
                correlations[i, j] = avg_similarity
                correlations[j, i] = avg_similarity
                
                logger.info(f"方法 {i+1} 与方法 {j+1} 的相关性: {avg_similarity:.4f}")
    
    return correlations

def process_word_folder(word_folder: str, base_hs_dir: str, methods: List[str], 
                       batch_size: int, file_pattern: str, output_dir: str,
                       layer_indices: List[int]) -> Tuple[np.ndarray, List[str]]:
    """处理单个词汇文件夹"""
    logger.info(f"======= 处理词汇: {word_folder} =======")
    
    # 收集方法路径和名称
    method_paths = []
    method_names = []
    
    for method in methods:
        # 为该方法找到最新的隐藏状态目录
        hs_dir_pattern = os.path.join(base_hs_dir, word_folder, method, "combined", "hidden_states_*")
        hs_dirs = glob.glob(hs_dir_pattern)
        
        if not hs_dirs:
            logger.warning(f"未找到方法 {method} 的隐藏状态目录")
            continue
        
        # 选择最新的目录
        hs_dir = max(hs_dirs, key=os.path.getmtime)
        method_paths.append(hs_dir)
        method_names.append(method)
    
    # 确保至少有两个方法
    if len(method_paths) < 2:
        logger.warning(f"词汇 {word_folder} 的有效方法少于2个，跳过")
        return None, None
    
    # 加载每个方法的数据
    method_data = []
    valid_method_names = []
    
    for folder, name in zip(method_paths, method_names):
        data = load_method_samples(folder, file_pattern)
        if data is not None and data.ndim == 4:
            method_data.append(data)
            valid_method_names.append(name)
            logger.info(f"加载了 {name} 的数据: {data.shape[0]} 样本, {data.shape[1]} 层, 第三维大小: {data.shape[2]}")
        else:
            if data is not None:
                logger.error(f"方法 {name} 的数据形状不正确: {data.shape}")
            else:
                logger.error(f"无法加载方法 {name} 的数据")
    
    # 确保至少有两个有效的方法数据
    if len(method_data) < 2:
        logger.warning(f"词汇 {word_folder} 至少需要两个有效的方法数据，跳过")
        return None, None
    
    # 创建输出目录
    word_output_dir = os.path.join(output_dir, word_folder)
    os.makedirs(word_output_dir, exist_ok=True)
    
    # 计算相关性矩阵
    logger.info(f"计算词汇 {word_folder} 的方法相关性")
    correlation_matrix = parallel_batch_compute_correlation(method_data, batch_size, layer_indices)
    
    # 绘制和保存相关性热图
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
    """聚合所有词汇的相关性矩阵"""
    logger.info("聚合所有词汇的相关性矩阵")
    
    if not all_correlations:
        logger.error("没有有效的相关性矩阵可以聚合")
        return
    
    # 计算平均相关性矩阵
    avg_correlation = np.mean(all_correlations, axis=0)
    
    # 保存聚合的相关性矩阵
    np.save(os.path.join(output_dir, "aggregate_correlation_matrix.npy"), avg_correlation)
    
    # 绘制和保存聚合的热图
    layers_str = ",".join(map(str, layer_indices))
    plot_correlation_heatmap(
        correlation_matrix=avg_correlation,
        method_names=method_names,
        title=f"Aggregate Method Correlation (Layers: {layers_str})",
        output_file=output_file
    )

def main():
    parser = argparse.ArgumentParser(description="计算不同提取方法之间的相关性")
    parser.add_argument("--base_dir", required=True, help="基础隐藏状态目录")
    parser.add_argument("--word_folders", required=True, help="要处理的词汇文件夹，用逗号分隔")
    parser.add_argument("--methods", required=True, help="要比较的提取方法，用逗号分隔")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=100, help="计算相关性的批次大小")
    parser.add_argument("--file_pattern", default="*.pt", help="隐藏状态文件模式")
    parser.add_argument("--layers", default="1", help="要使用的层索引，用逗号分隔（0-based）")
    
    args = parser.parse_args()
    
    # 解析层索引
    layer_indices = [int(idx) for idx in args.layers.split(",")]
    
    # 解析词汇文件夹和方法
    word_folders = args.word_folders.split(",")
    methods = args.methods.split(",")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个词汇文件夹
    all_correlations = []
    common_method_names = None
    
    for folder in word_folders:
        correlation_matrix, method_names = process_word_folder(
            folder, args.base_dir, methods, args.batch_size, 
            args.file_pattern, args.output_dir, layer_indices
        )
        
        if correlation_matrix is not None:
            all_correlations.append(correlation_matrix)
            
            # 保存第一个有效的方法名列表作为通用方法名
            if common_method_names is None:
                common_method_names = method_names
    
    # 如果有有效的相关性矩阵，聚合它们
    if all_correlations and common_method_names:
        aggregate_output_file = os.path.join(args.output_dir, "aggregate_method_correlation.png")
        aggregate_correlations(
            all_correlations, common_method_names, 
            aggregate_output_file, args.output_dir, layer_indices
        )
    else:
        logger.error("没有有效的相关性矩阵可以聚合")

if __name__ == "__main__":
    main() 