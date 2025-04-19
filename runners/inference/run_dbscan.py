import argparse
import os


from evaluate import merge_label, resolve_file_path
from plot import plot_hidden_states_layers_spontaneous_clusters, fix_tensor_from_series
from logger_config import setup_logger

# 设置日志
logger = setup_logger('run_dbscan', 'logs/run_dbscan.log')

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="使用DBSCAN对多层隐藏状态进行聚类分析")
    parser.add_argument('--hidden_states_dir', type=str, required=True, help='隐藏状态张量文件路径(.pt)')
    parser.add_argument('--sentence_csv', type=str, required=True, help='句子CSV文件路径')
    #parser.add_argument('--label_csv', type=str, required=True, help='标签CSV文件路径') 
    parser.add_argument('--output_dir', type=str, default='layers_dbscan_plots', help='图表保存目录')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN的eps参数')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN的min_samples参数')
    parser.add_argument('--method', type=str, default='umap', choices=['pca', 'tsne', 'umap', 'zca_pca', 'zca_tsne', 'zca_umap'], help='降维方法')
    args = parser.parse_args()

    # 找到隐藏状态文件
    hidden_states_file = resolve_file_path(
        args.hidden_states_dir,
        "*.pt", 
        f"错误: 隐藏状态文件未找到: {args.hidden_states_dir}",
        logger
    )
    
    if hidden_states_file is None:
        return
        
    # 检查句子CSV文件是否存在
    if not os.path.exists(args.sentence_csv):
        logger.error(f"错误: 句子CSV文件未找到: {args.sentence_csv}")
        return

    # 合并数据
    logger.info("开始合并数据...")
    df_merged = merge_label(
        hs_pt_file=hidden_states_file,
        sentence_csv_file=args.sentence_csv,
        label_csv_file=None
    )
     
    if df_merged.empty:
        logger.error("错误: 合并后的数据为空")
        return

    logger.info(f"数据合并完成，共 {len(df_merged)} 行")

    # 提取隐藏状态
    hidden_states = fix_tensor_from_series(df_merged["hidden_states"])
    logger.info(f"隐藏状态张量形状: {hidden_states.shape}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 执行DBSCAN聚类并绘图
    logger.info(f"开始使用DBSCAN聚类分析，eps={args.eps}, min_samples={args.min_samples}")
    plot_hidden_states_layers_spontaneous_clusters(
        hidden_states,
        df=df_merged,
        content_column='generated_content',
        eps=args.eps,
        min_samples=args.min_samples,
        method=args.method,
        save_dir=args.output_dir,
        show_plots=False
    )

    logger.info(f"聚类分析完成，图表已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
