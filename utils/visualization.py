# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
import seaborn as sns
from torchvision.utils import make_grid


class Visualizer:
    def __init__(self, save_dir):
        """
        初始化可视化器
        Args:
            save_dir: 保存可视化结果的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 创建子目录
        self.attention_dir = os.path.join(save_dir, 'attention_maps')
        self.prediction_dir = os.path.join(save_dir, 'predictions')
        self.metrics_dir = os.path.join(save_dir, 'metrics')

        for d in [self.attention_dir, self.prediction_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

    def visualize_prediction(self, image, mask, prediction, index, epoch=None):
        """
        可视化单个预测结果
        Args:
            image: 原始图像 (C,H,W)
            mask: 真实掩码 (H,W)
            prediction: 预测掩码 (H,W)
            index: 样本索引
            epoch: 训练轮次（可选）
        """
        plt.figure(figsize=(15, 5))

        # 显示原始图像
        plt.subplot(131)
        img_np = image.transpose(1, 2, 0)  # 转换为(H,W,C)
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')

        # 显示真实掩码
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # 显示预测结果
        plt.subplot(133)
        plt.imshow(prediction > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        # 保存图像
        filename = f'pred_{index}.png' if epoch is None else f'epoch_{epoch}_pred_{index}.png'
        plt.savefig(os.path.join(self.prediction_dir, filename))
        plt.close()

    def visualize_attention(self, attention_maps, index, epoch=None):
        """
        可视化注意力图
        Args:
            attention_maps: 注意力特征图 (B,C,H,W)
            index: 样本索引
            epoch: 训练轮次（可选）
        """
        if torch.is_tensor(attention_maps):
            attention_maps = attention_maps.cpu().detach().numpy()

        # 计算平均注意力图
        avg_attention = np.mean(attention_maps, axis=1)  # (B,H,W)

        for i, att in enumerate(avg_attention):
            plt.figure(figsize=(6, 6))
            sns.heatmap(att, cmap='jet', cbar=True)
            plt.title(f'Attention Map {index}_{i}')

            # 保存图像
            filename = f'attention_{index}_{i}.png' if epoch is None else f'epoch_{epoch}_attention_{index}_{i}.png'
            plt.savefig(os.path.join(self.attention_dir, filename))
            plt.close()

    def plot_metrics(self, metrics_history, current_epoch):
        """
        绘制训练指标曲线
        Args:
            metrics_history: 包含训练历史的字典
            current_epoch: 当前训练轮次
        """
        plt.figure(figsize=(12, 8))

        # 创建轮次数组
        epochs = list(range(1, current_epoch + 1))

        # 为每个指标创建子图
        num_metrics = len(metrics_history)
        for i, (metric_name, metric_values) in enumerate(metrics_history.items(), 1):
            plt.subplot(num_metrics, 1, i)

            # 确保数据长度匹配
            values = metric_values[:len(epochs)]

            # 绘制曲线
            plt.plot(epochs, values, 'b-', label=metric_name)
            plt.title(f'{metric_name} vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric_name)
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, f'metrics_epoch_{current_epoch}.png'))
        plt.close()

    def visualize_batch(self, images, masks, predictions, attention_maps, epoch, batch_idx):
        """
        可视化一个批次的结果
        Args:
            images: 批次图像 (B,C,H,W) - tensor
            masks: 真实掩码 (B,H,W) - tensor
            predictions: 预测掩码 (B,H,W) - tensor
            attention_maps: 注意力图 (B,C,H,W) - tensor
            epoch: 训练轮次
            batch_idx: 批次索引
        """
        # 确保所有输入都是numpy数组
        images_np = images.detach().cpu().numpy() if torch.is_tensor(images) else images
        masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else masks
        predictions_np = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else predictions
        attention_maps_np = attention_maps.detach().cpu().numpy() if torch.is_tensor(attention_maps) else attention_maps

        # 可视化预测结果
        for i in range(min(images_np.shape[0], 4)):  # 最多显示4个样本
            self.visualize_prediction(
                images_np[i],
                masks_np[i],
                predictions_np[i],
                f'{batch_idx}_{i}',
                epoch
            )

        # 可视化注意力图
        self.visualize_attention(attention_maps_np, batch_idx, epoch)

     #此处加入对比图
    # 图片比较对比图
    def save_comparison_grid(self, images, masks, predictions, save_path):
        """
        将单个样本的原图、掩码和预测结果保存为对比图
        Args:
            images: 图像张量 (1,C,H,W)
            masks: 掩码张量 (1,H,W)
            predictions: 预测张量 (1,H,W)
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 5))

        # 显示原始图像
        plt.subplot(131)
        img_np = images[0].numpy().transpose(1, 2, 0)
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')

        # 显示真实掩码
        plt.subplot(132)
        plt.imshow(masks[0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # 显示预测结果
        plt.subplot(133)
        plt.imshow(predictions[0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def save_training_process(save_dir, epoch, model, optimizer, loss, metrics):
    """
    保存训练过程信息
    Args:
        save_dir: 保存目录
        epoch: 当前轮次
        model: 模型
        optimizer: 优化器
        loss: 损失值
        metrics: 评估指标
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }

    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))


def load_training_process(save_dir, epoch, model, optimizer):
    """
    加载训练过程信息
    Args:
        save_dir: 保存目录
        epoch: 要加载的轮次
        model: 模型
        optimizer: 优化器
    Returns:
        epoch: 加载的轮次
        loss: 损失值
        metrics: 评估指标
    """
    checkpoint = torch.load(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['loss'], checkpoint['metrics']


class MetricsVisualizer:
    def __init__(self, save_dir):
        """
        初始化指标可视化器
        Args:
            save_dir: 保存可视化结果的目录
        """
        self.metrics_dir = os.path.join(save_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_metric_train(self, epochs, metric_values, metric_name):
        """
        为单个指标绘制折线图
        Args:
            epochs: epoch数组
            metric_values: 指标值数组
            metric_name: 指标名称
        """
        plt.figure(figsize=(12, 8))

        # 绘制主曲线
        plt.plot(epochs, metric_values, 'b-', linewidth=2, label=metric_name)

        # 添加平滑曲线
        if len(metric_values) > 5:  # 只在数据点足够时添加平滑曲线
            smooth_values = self._smooth_curve(metric_values)
            plt.plot(epochs, smooth_values, 'r--', linewidth=1, label=f'Smoothed {metric_name}')

        # 设置图表属性
        plt.title(f'Training {metric_name} Over Time', fontsize=14, pad=20)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 添加最大值和最小值标注
        max_value = max(metric_values)
        min_value = min(metric_values)
        max_epoch = epochs[np.argmax(metric_values)]
        min_epoch = epochs[np.argmin(metric_values)]

        plt.annotate(f'Max: {max_value:.4f}',
                     xy=(max_epoch, max_value),
                     xytext=(10, 10),
                     textcoords='offset points',
                     ha='left',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.annotate(f'Min: {min_value:.4f}',
                     xy=(min_epoch, min_value),
                     xytext=(10, -10),
                     textcoords='offset points',
                     ha='left',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # 保存图表
        plt.tight_layout()
        save_path = os.path.join(self.metrics_dir, f'{metric_name.lower()}_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    def plot_metric_test(self, x_values, metric_values, metric_name):
        """
        为单个指标生成详细的折线图
        Args:
            x_values: x轴值（batch数或epoch数）
            metric_values: 指标值列表
            metric_name: 指标名称
        """
        plt.figure(figsize=(12, 8))

        # 绘制主曲线
        plt.plot(x_values, metric_values, 'b-', linewidth=2, label=f'{metric_name} per batch')

        # 计算移动平均线
        window_size = min(len(metric_values) // 10, 20)  # 动态窗口大小
        if window_size > 1:
            smoothed = np.convolve(metric_values, np.ones(window_size) / window_size, mode='valid')
            plt.plot(x_values[window_size - 1:], smoothed, 'r--',
                     linewidth=1.5, label=f'Moving average (window={window_size})')

        # 添加统计信息
        mean_value = np.mean(metric_values)
        std_value = np.std(metric_values)
        max_value = np.max(metric_values)
        min_value = np.min(metric_values)

        # 在图表上添加统计信息
        info_text = (f'Mean: {mean_value:.4f}\n'
                     f'Std: {std_value:.4f}\n'
                     f'Max: {max_value:.4f}\n'
                     f'Min: {min_value:.4f}')

        plt.text(0.02, 0.98, info_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 设置图表属性
        plt.title(f'{metric_name} Evolution During Evaluation', fontsize=14, pad=20)
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='lower right')
        # 保存图表
        save_path = os.path.join(self.metrics_dir, f'{metric_name.lower()}_evolution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_all_metrics(self, metrics_history):
        """
        为所有指标生成可视化图表
        Args:
            metrics_history: 包含所有指标历史数据的字典
        """
        epochs = list(range(1, len(next(iter(metrics_history.values()))) + 1))
        
        # 为每个指标生成单独的图表
        for metric_name, metric_values in metrics_history.items():
            self.plot_metric_train(epochs, metric_values, metric_name)

    def _smooth_curve(self, points, factor=0.8):
        """
        使用指数移动平均平滑曲线
        """
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points



