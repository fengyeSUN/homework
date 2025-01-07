# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import yaml
import os
from torch.utils.data import DataLoader
from data.dataset_origion import SmallObjectDataset
from models.small_object_net import SmallObjectNet
from utils.metrics_old import evaluate_batch, evaluate_model_metrics
from utils.visualization import Visualizer
from tqdm import tqdm
import logging
import numpy as np
from datetime import datetime
from utils.visualization import MetricsVisualizer  # 更新导入


def setup_logging(save_dir):
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'evaluation.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def format_metrics_for_yaml(metrics):
    """格式化指标为易读格式"""
    return {k: float(v) for k, v in metrics.items()}

def evaluate_model(config, model_path):
    """
    评估模型性能
    Args:
        config: 配置字典
        model_path: 模型权重文件路径
    Returns:
        dict: 评估指标
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 创建保存目录
    save_dir = os.path.join(config['train']['save_dir'], 'evaluation')
    os.makedirs(save_dir, exist_ok=True)
    predictions_dir = os.path.join(save_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # 设置日志
    setup_logging(save_dir)

    # 创建数据加载器
    test_dataset = SmallObjectDataset(
        root_dir=config['data']['test_dir'],
        split='test'
    )
    logging.info(f"Test dataset size: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    # 加载模型
    model = SmallObjectNet(
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim']
    ).to(device)

    # 加载模型权重
    logging.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
    else:
        model.load_state_dict(checkpoint)
        epoch = 0
    model.eval()

    # 创建可视化器
    visualizer = Visualizer(save_dir)

    # 初始化指标累加器
    batch_metrics_sum = {}
    all_predictions = []
    all_targets = []
    # 添加batch级别的指标跟踪
    batch_metrics_history = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'dice': []
    }
    # 评估
    logging.info("Starting evaluation...")
    with torch.no_grad():
        with tqdm(test_loader, desc='Evaluating') as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)

                # 调整掩码维度
                masks = masks.squeeze(1)

                # 前向传播
                outputs, attention_maps = model(images)
                predictions = torch.sigmoid(outputs)

                # 计算批次指标
                batch_metrics = evaluate_batch(
                    predictions.cpu().numpy(),
                    masks.cpu().numpy()
                )
                # 更新指标历史
                for metric_name, value in batch_metrics.items():
                    batch_metrics_history[metric_name].append(value)

                # 更新指标累加器
                for k, v in batch_metrics.items():
                    batch_metrics_sum[k] = batch_metrics_sum.get(k, 0) + v

                # 收集预测结果用于整体评估
                all_predictions.extend(predictions.cpu().numpy().reshape(-1))
                all_targets.extend(masks.cpu().numpy().reshape(-1))

                # 更新进度条
                current_metrics = {k: v / (batch_idx + 1) for k, v in batch_metrics_sum.items()}
                pbar.set_postfix(current_metrics)

                # 保存每个测试样本的预测结果
                for i in range(images.size(0)):
                    img_idx = batch_idx * test_loader.batch_size + i
                    if img_idx < len(test_dataset):  # 确保不超过数据集大小
                        sample_name = test_dataset.images[img_idx].split('.')[0]
                        visualizer.save_comparison_grid(
                            images=images[i:i + 1].cpu(),
                            masks=masks[i:i + 1].cpu(),
                            predictions=predictions[i:i + 1].cpu(),
                            save_path=os.path.join(predictions_dir, f'{sample_name}_comparison.png')
                        )

                # 可视化注意力图和批次结果
                if batch_idx % config['visualization']['frequency'] == 0:
                    visualizer.visualize_batch(
                        images=images.cpu(),
                        masks=masks.cpu(),
                        predictions=predictions.cpu(),
                        attention_maps=attention_maps.cpu(),
                        batch_idx=batch_idx,
                        epoch=epoch
                    )

    # 计算并记录最终指标
    overall_metrics = evaluate_model_metrics(all_predictions, all_targets)

    # 创建MetricsVisualizer实例
    metrics_visualizer = MetricsVisualizer(save_dir)

    # 生成每个指标的可视化图表
    batch_numbers = list(range(1, len(batch_metrics_history['accuracy']) + 1))
    for metric_name, values in batch_metrics_history.items():
        metrics_visualizer.plot_metric_test(
            batch_numbers,
            values,
            f'Batch {metric_name}'
        )

    # 计算平均批次指标
    avg_batch_metrics = {k: v / len(test_loader) for k, v in batch_metrics_sum.items()}

    # 计算整体指标
    overall_metrics = evaluate_model_metrics(all_predictions, all_targets)

    # 保存评估结果
    '''results = {
        'batch_average_metrics': avg_batch_metrics,
        'overall_metrics': overall_metrics,
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path
    }'''
    results = {
        'batch_average_metrics': format_metrics_for_yaml(avg_batch_metrics),
        'overall_metrics': format_metrics_for_yaml(overall_metrics),
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path
    }

    # 记录评估结果（格式化输出）
    logging.info("\n" + "=" * 50)
    logging.info("Evaluation Results:")
    logging.info("=" * 50)

    logging.info("\nBatch Average Metrics:")
    logging.info("-" * 30)
    metric_format = "{:<15} {:<10.4f}"
    for metric_name, value in avg_batch_metrics.items():
        logging.info(metric_format.format(metric_name, value))

    logging.info("\nOverall Metrics:")
    logging.info("-" * 30)
    for metric_name, value in overall_metrics.items():
        logging.info(metric_format.format(metric_name, value))

    logging.info("\n" + "=" * 50)

    # 保存结果到文件
    results_file = os.path.join(save_dir, 'evaluation_results.yaml')
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, allow_unicode=True)
    logging.info(f"\nDetailed results saved to: {results_file}")
    logging.info(f"Predictions saved to: {predictions_dir}")

    return overall_metrics


if __name__ == '__main__':
    # 加载配置
    try:
        with open('configs/config.yaml', 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open('configs/config.yaml', 'r', encoding='gbk') as f:
            config = yaml.safe_load(f)

    # 使用最佳模型进行评估
    model_path = os.path.join(config['train']['save_dir'], 'best_model.pth')
    if not os.path.exists(model_path):
        model_files = [f for f in os.listdir(config['train']['save_dir'])
                       if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError("No model files found!")
        model_path = os.path.join(config['train']['save_dir'], sorted(model_files)[-1])

    logging.info(f"Using model: {model_path}")
    metrics = evaluate_model(config, model_path)