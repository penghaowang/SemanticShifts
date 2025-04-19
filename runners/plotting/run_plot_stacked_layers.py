import argparse
import torch
import os
import numpy as np
import logging
import matplotlib.pyplot as plt

# 从 evaluate.py 导入已有函数
from evaluate import merge_label, resolve_file_path, reduce_dimensions
from plot import fix_tensor_from_series
from logger_config import setup_logger

# 设置日志
logger = setup_logger('run_plot_stacked_layers', 'logs/run_plot_stacked_layers.log')

def plot_stacked_layers(
    hidden_states,
    labels=None,
    method='umap',
    title='Stacked Layers',
    save_path=None,
    show_plot=False,
    df=None,
    content_column='generated_content',
    top_n=3
):
    """
    将多层隐藏状态堆叠，降维后绘制一张散点图。
    
    参数:
        hidden_states (torch.Tensor/np.ndarray): [samples, features] 已堆叠的隐藏状态
        labels (list/ndarray): 每个样本对应的标签(长度=samples)，若为 None，则不区分颜色。
        method (str): 降维方法，默认 'umap'。
        title (str): 图表标题
        save_path (str): 保存图表的路径
        show_plot (bool): 是否显示图表
        df (pandas.DataFrame, optional): 包含原始文本内容的数据帧，用于统计每个标签最常见内容。
        content_column (str): df 中包含文本内容的列名，默认为 'generated_content'。
        top_n (int): 为每个标签显示的最常见内容数量，默认为 3。
    """
    # 转 NumPy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    elif not isinstance(hidden_states, np.ndarray):
        raise ValueError("hidden_states 必须是 PyTorch 张量或 NumPy 数组。")

    # 降维到2D
    reduced_vectors = reduce_dimensions(hidden_states, method=method, n_components=2)

    plt.figure(figsize=(8, 6))

    # 如果没提供 labels，就统一用一个颜色
    if labels is None:
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7, c='blue')
    else:
        # 有标签，按每个唯一标签分别绘制散点，并带图例
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap('rainbow', len(unique_labels))

        for i, lab in enumerate(unique_labels):
            idx = [j for j, x in enumerate(labels) if x == lab]
            plt.scatter(
                reduced_vectors[idx, 0],
                reduced_vectors[idx, 1],
                color=cmap(i),
                label=str(lab),
                alpha=0.7
            )
        plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    # 打印每个 label 最常见的内容（如果提供了df）
    if labels is not None and df is not None and content_column in df.columns:
        if len(df) == len(labels):
            from collections import Counter
            footer_texts = []
            unique_labels = sorted(set(labels))
            for lab in unique_labels:
                # 筛选出该标签对应的行
                mask = (labels == lab)
                sub_df = df[mask]
                # 统计最常见的 top_n
                content_counts = Counter(sub_df[content_column])
                top_items = content_counts.most_common(top_n)
                if not top_items:
                    footer_texts.append(f"Label {lab}: (无内容)")
                else:
                    # 拼一个简短的字符串
                    items_str = " | ".join(
                        f"{txt}({cnt}x)" for txt, cnt in top_items
                    )
                    footer_texts.append(f"Label {lab}: {items_str}")

            # 将这些统计文字放在图的下方
            text_block = "\n".join(footer_texts)
            plt.subplots_adjust(bottom=0.3)
            plt.figtext(0.01, 0.01, text_block, fontsize=8, va="bottom", ha="left")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"图表已保存到 {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def stack_layer_group(hidden_states, start_layer, end_layer):
    """
    将指定范围内的层的隐藏状态堆叠到一起
    
    参数:
        hidden_states (torch.Tensor/np.ndarray): [samples, layers, hidden_dim] 或 [samples, layers, beams, hidden_dim]
        start_layer (int): 开始层索引（从0开始）
        end_layer (int): 结束层索引（不包含）
        
    返回:
        堆叠后的隐藏状态 [samples, (end_layer-start_layer)*hidden_dim]
    """
    # 转 NumPy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    
    # 如果是四维 [samples, layers, beams, hidden_dim]，先取beam=0
    if hidden_states.ndim == 4:
        hidden_states = hidden_states[:, :, 0, :]
    
    # 提取指定范围的层
    layers_subset = hidden_states[:, start_layer:end_layer, :]  # [samples, selected_layers, hidden_dim]
    
    # 获取形状信息
    samples, selected_layers, hidden_dim = layers_subset.shape
    
    # 重塑张量并堆叠层
    # 从 [samples, selected_layers, hidden_dim] 变为 [samples, selected_layers*hidden_dim]
    stacked_features = layers_subset.reshape(samples, selected_layers * hidden_dim)
    
    return stacked_features

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="将层组隐藏状态堆叠并绘制散点图")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='隐藏状态张量文件路径 (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=False, help='句子CSV文件路径')
    parser.add_argument('--label_csv', type=str, required=False, help='标签CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='stacked_layers_output', help='图表保存目录')
    parser.add_argument('--method', type=str, default='umap', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='降维方法')
    args = parser.parse_args()

    # 查找隐藏状态文件
    hidden_states_file = resolve_file_path(
        args.hidden_states_dir,
        "*.pt", 
        f"未找到隐藏状态文件: {args.hidden_states_dir}",
        logger
    )
    
    if hidden_states_file is None:
        return

    # 加载隐藏状态数据
    hidden_states = torch.load(hidden_states_file)
    logger.info(f"已加载隐藏状态，形状为: {hidden_states.shape}")
    
    # 定义三个层组
    layer_groups = [
        {"start": 0, "end": 10, "name": "Early Layers (1-10)"},
        {"start": 10, "end": 20, "name": "Middle Layers (11-20)"},
        {"start": 20, "end": 33, "name": "Later Layers (21-33)"}
    ]

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 如果没有提供 sentence_csv 和 label_csv，则仅绘制隐藏状态(不区分标签)
    if not args.sentence_csv or not args.label_csv:
        logger.info("未提供 sentence_csv 和 label_csv，将只绘制未带标签的散点图。")
        
        # 为每个层组生成堆叠图表
        for group in layer_groups:
            # 堆叠该组的所有层
            stacked_features = stack_layer_group(
                hidden_states, 
                start_layer=group["start"], 
                end_layer=group["end"]
            )
            
            # 生成图表
            save_path = os.path.join(args.output_dir, f"stacked_{group['name'].replace(' ', '_').lower()}.png")
            plot_stacked_layers(
                stacked_features,
                labels=None,  # 不区分颜色
                method=args.method,
                title=f"{group['name']} - Stacked ({args.method.upper()})",
                save_path=save_path,
                show_plot=False
            )
            
        logger.info(f"所有层组堆叠图表已保存到目录: {args.output_dir}")
    else:
        # 如果提供了 sentence_csv 和 label_csv，则执行 merge_label
        logger.info("提供了 sentence_csv 和 label_csv，开始合并并绘制带标签的散点图。")
        df_merged = merge_label(
            hs_pt_file=hidden_states_file,
            sentence_csv_file=args.sentence_csv,
            label_csv_file=args.label_csv
        )
        if df_merged.empty:
            logger.error("合并结果为空，无法绘图。")
            return

        logger.info(f"合并完成，DataFrame 行数: {len(df_merged)}")
        # 从df_merged中提取隐藏状态和标签，使用fix_tensor_from_series避免警告
        hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
        logger.debug(f"hidden_states shape: {hidden_states.shape}")
        labels = df_merged["definition"]  # 使用definition列作为标签

        # 为每个层组生成堆叠图表
        for group in layer_groups:
            # 堆叠该组的所有层
            stacked_features = stack_layer_group(
                hidden_states, 
                start_layer=group["start"], 
                end_layer=group["end"]
            )
            
            # 生成图表
            save_path = os.path.join(args.output_dir, f"stacked_{group['name'].replace(' ', '_').lower()}.png")
            plot_stacked_layers(
                stacked_features,
                labels=labels,
                method=args.method,
                title=f"{group['name']} - Stacked ({args.method.upper()})",
                save_path=save_path,
                show_plot=False,
                df=df_merged,
                content_column='generated_content',
                top_n=3
            )
            
        logger.info(f"所有层组堆叠图表已保存到目录: {args.output_dir}")

if __name__ == "__main__":
    main() 