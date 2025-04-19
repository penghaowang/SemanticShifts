import argparse
import torch
import os
import numpy as np
import logging

# 从 evaluate.py 导入刚才写好的函数
from evaluate import merge_label, resolve_file_path
from plot import plot_hidden_states, fix_tensor_from_series
from logger_config import setup_logger

# 设置日志
logger = setup_logger('run_plot_layer_groups', 'logs/run_plot_layer_groups.log')

def plot_hidden_states_layer_group(
    hidden_states,
    labels=None,
    method='pca',
    save_dir='.',
    show_plots=False,
    df=None,
    content_column='generated_content',
    top_n=3,
    start_layer=0,
    end_layer=None,
    group_name=""
):
    """
    对指定范围层的隐藏状态分别降维并绘制散点图（可选标签）。
    
    参数:
        hidden_states (torch.Tensor/np.ndarray): [samples, layers, beams, hidden_dim] 或 [samples, layers, hidden_dim]
        labels (list/ndarray): 每个样本对应的标签(长度=samples)，若为 None，则不区分颜色。
        method (str): 降维方法，默认 'pca'。
        save_dir (str): 保存图表的目录路径。
        show_plots (bool): 是否在函数结束后调用 plt.show()。
        df (pandas.DataFrame, optional): 包含原始文本内容的数据帧，用于统计每个标签最常见内容。
        content_column (str): df 中包含文本内容的列名，默认为 'generated_content'。
        top_n (int): 为每个标签显示的最常见内容数量，默认为 3。
        start_layer (int): 开始层索引（从0开始）
        end_layer (int): 结束层索引（不包含），如为None则绘制到最后一层
        group_name (str): 层组的名称，用于标题和文件名
    """
    # 转 NumPy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    elif not isinstance(hidden_states, np.ndarray):
        raise ValueError("hidden_states 必须是 PyTorch 张量或 NumPy 数组。")

    if hidden_states.ndim not in [3,4]:
        raise ValueError("hidden_states 维度必须是 [N, L, D] 或 [N, L, B, D].")

    # 如果是四维 [samples, layers, beams, hidden_dim]，先根据 beams 取一个
    if hidden_states.ndim == 4:
        # 例如只取 beam=0
        hidden_states = hidden_states[:, :, 0, :]

    # 现在是 [samples, layers, hidden_dim]
    num_layers = hidden_states.shape[1]
    
    # 确定结束层索引
    if end_layer is None or end_layer > num_layers:
        end_layer = num_layers
    
    # 创建保存目录
    group_dir = os.path.join(save_dir, f"layers_{start_layer+1}_to_{end_layer}")
    os.makedirs(group_dir, exist_ok=True)
    
    logger.info(f"开始绘制层 {start_layer+1} 到 {end_layer} 的图表 ({group_name})")

    # 只绘制指定范围内的层
    for layer_idx in range(start_layer, end_layer):
        layer_hs = hidden_states[:, layer_idx, :]  # [samples, hidden_dim]
        title = f'{group_name} - Layer {layer_idx+1} Scatter ({method.upper()})'
        save_path = os.path.join(group_dir, f'layer_{layer_idx+1}_{method}_scatter.png')

        plot_hidden_states(
            layer_hs,
            labels=labels,
            method=method,
            title=title,
            save_path=save_path,
            show_plot=show_plots,
            df=df,
            content_column=content_column,
            top_n=top_n
        )
    
    logger.info(f"层组 {group_name} 的图表已保存到 {group_dir}")


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="绘制指定层组的隐藏状态散点图")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='隐藏状态张量文件路径 (.pt)')
    parser.add_argument('--sentence_csv', type=str, required=False, help='句子CSV文件路径')
    parser.add_argument('--label_csv', type=str, required=False, help='标签CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='layer_groups_output', help='图表保存目录')
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

    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 如果没有提供 sentence_csv 和 label_csv，则仅绘制隐藏状态(不区分标签)
    if not args.sentence_csv or not args.label_csv:
        logger.info("未提供 sentence_csv 和 label_csv，将只绘制未带标签的散点图。")
        
        # 为每个层组生成图表
        for group in layer_groups:
            plot_hidden_states_layer_group(
                hidden_states, 
                labels=None,  # 不区分颜色
                method=args.method,
                save_dir=args.output_dir, 
                show_plots=False,
                start_layer=group["start"],
                end_layer=group["end"],
                group_name=group["name"]
            )
            
        logger.info(f"所有层组图表已保存到目录: {args.output_dir}")
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

        # 为每个层组生成图表
        for group in layer_groups:
            plot_hidden_states_layer_group(
                hidden_states, 
                labels=labels, 
                method=args.method, 
                save_dir=args.output_dir, 
                show_plots=False,
                df=df_merged,
                content_column='generated_content',
                top_n=3,
                start_layer=group["start"],
                end_layer=group["end"],
                group_name=group["name"]
            )
            
        logger.info(f"所有层组图表已保存到目录: {args.output_dir}")

if __name__ == "__main__":
    main() 