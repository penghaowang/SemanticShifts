import argparse
import os
import glob
import torch
import numpy as np
from evaluate import cosine_similarity_layers, resolve_file_path
from plot import plot_heatmap_layerwise, plot_relative_similarity
from logger_config import setup_logger



# 设置模块特定的日志记录器
logger = setup_logger('run_layer_plot', 'logs/run_layer_plot.log')

#先对每一个词做这两个plot 然后对整体做这两个plot

def main():
    parser = argparse.ArgumentParser(description='Layer-wise plot')
    parser.add_argument('--base_hs_dir', type=str, required=True, help='Base directory containing all hidden states')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--method', type=str, required=True, help='Method name (e.g., input_last_token, eos_token)')
    args = parser.parse_args()

    logger.info(f"开始处理层间相似度分析，基础目录: {args.base_hs_dir}，方法: {args.method}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录已创建: {args.output_dir}")

    # 查找所有包含该方法的隐藏状态目录
    word_dirs_pattern = os.path.join(args.base_hs_dir, "*", args.method, "combined", "hidden_states_*")
    word_dirs = glob.glob(word_dirs_pattern)
    
    if not word_dirs:
        logger.error(f"未找到任何匹配的隐藏状态目录: {word_dirs_pattern}")
        return
    
    logger.info(f"找到 {len(word_dirs)} 个词汇目录")
    
    # 创建一个字典来存储每个词的层相似度
    word_layer_similarities = {}
    all_layer_similarities = []
    
    # 遍历每个词汇目录
    for word_dir in word_dirs:
        # 从目录路径中提取词汇名
        word_folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(word_dir))))
        logger.info(f"处理词汇目录: {word_folder}")
        
        # 查找该目录中的所有隐藏状态文件
        hidden_states_files = glob.glob(os.path.join(word_dir, "*.pt"))
        
        if not hidden_states_files:
            logger.warning(f"词汇目录 '{word_folder}' 中未找到隐藏状态文件，跳过")
            continue
        
        logger.info(f"词汇 {word_folder} 找到 {len(hidden_states_files)} 个文件")
        
        # 遍历该词汇的所有隐藏状态文件
        for file in hidden_states_files:
            logger.info(f"处理文件: {file}")
            
            # 加载隐藏状态
            try:
                hidden_states = torch.load(file)
                logger.debug(f"加载隐藏状态，形状: {hidden_states.shape}")
                
                # 计算层级相似度
                logger.info(f"计算 {word_folder} 的层间相似度")
                layer_similarity = cosine_similarity_layers(hidden_states)
                
                # 存储该词的层相似度
                word_key = f"{word_folder}_{os.path.basename(file)}"
                word_layer_similarities[word_key] = layer_similarity.cpu().numpy() if isinstance(layer_similarity, torch.Tensor) else layer_similarity
                
                # 添加到总的层相似度列表
                all_layer_similarities.append(layer_similarity)
                
                # 绘制层级热图
                heatmap_path = os.path.join(args.output_dir, f'{word_folder}_{os.path.basename(file)}_heatmap.png')
                logger.info(f"绘制 {word_folder} 的层级热图，保存至: {heatmap_path}")
                plot_heatmap_layerwise(layer_similarity, heatmap_path)
                
                # 绘制相对相似度热图
                rel_similarity_path = os.path.join(args.output_dir, f'{word_folder}_{os.path.basename(file)}_rel_similarity.png')
                logger.info(f"绘制 {word_folder} 的相对相似度图，保存至: {rel_similarity_path}")
                plot_relative_similarity(layer_similarity, rel_similarity_path)
            except Exception as e:
                logger.error(f"处理文件 {file} 时出错: {e}")
    
    # 如果有处理成功的词汇，则绘制总的图表
    if all_layer_similarities:
        logger.info("开始绘制所有词汇的综合层间相似度图")
        
        # 合并所有层相似度
        try:
            all_layer_similarity = np.concatenate([sim.cpu().numpy() if isinstance(sim, torch.Tensor) else sim for sim in all_layer_similarities], axis=0)
            logger.debug(f"合并后的层间相似度数组形状: {all_layer_similarity.shape}")
            
            # 绘制层级热图
            heatmap_path = os.path.join(args.output_dir, 'all_words_layer_heatmap.png')
            logger.info(f"绘制所有词汇的层级热图，保存至: {heatmap_path}")
            plot_heatmap_layerwise(all_layer_similarity, heatmap_path)
            
            # 绘制相对相似度热图
            rel_similarity_path = os.path.join(args.output_dir, 'all_words_rel_similarity.png')
            logger.info(f"绘制所有词汇的相对相似度图，保存至: {rel_similarity_path}")
            plot_relative_similarity(all_layer_similarity, rel_similarity_path)
        except Exception as e:
            logger.error(f"合并层相似度数组失败: {e}")
    else:
        logger.warning("没有成功处理任何词汇，无法绘制综合图表")
    
    logger.info("层间相似度分析完成")

if __name__ == "__main__":
    main()
