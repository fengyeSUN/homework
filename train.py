# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset_origion import SmallObjectDataset
from models.small_object_net import SmallObjectNet
from utils.visualization import Visualizer, save_training_process
from utils.metrics_old import evaluate_batch
import yaml
import os
from tqdm import tqdm
import logging
from utils.visualization import MetricsVisualizer  # 更新导入


def setup_logging(save_dir):
    """设置日志"""
    log_file = os.path.join(save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

class WeightedLoss(nn.Module):
    def __init__(self, weights=[0.4, 0.4, 0.2], pos_weight=30.0):#增加正样本权重#def __init__(self, weights=[0.3, 0.4, 0.3], pos_weight=10.0):
        super().__init__()
        self.weights = weights
        """self.focal = FocalLoss()
        self.dice = DiceLoss()"""
        self.focal = FocalLoss(alpha=0.75, gamma=2)  # 调整focal loss参数
        self.dice = DiceLoss(smooth=1e-6)  # 减小平滑因子
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    def forward(self, predictions, targets):
        focal_loss = self.focal(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        bce_loss = self.bce(predictions, targets)
        return self.weights[0] * focal_loss + self.weights[1] * dice_loss + self.weights[2] * bce_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(config):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(save_dir)
    
    # 创建数据加载器
    train_dataset = SmallObjectDataset(
        root_dir=config['data']['train_dir'],
        split='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    # 创建模型
    model = SmallObjectNet(
        num_classes=1,
        hidden_dim=config['model']['hidden_dim']
    ).to(device)
    
    # 定义损失函数
    criterion = WeightedLoss(
        weights=list(config['train']['loss_weights'].values()),
        pos_weight=config['train']['pos_weight']
    ).to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay'],
        betas=(0.9,0.999), #新增调整动量参数以期改善效果，新增
        eps=1e-8  #新增
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=8, #原来为5，尝试增加耐心值至8
        verbose=True,
        min_lr=1e-6 #设置最小学习率（尝试改善新增）
    )

    # 创建新的可视化器
    metrics_visualizer = MetricsVisualizer(save_dir)

    # 创建可视化器
    visualizer = Visualizer(save_dir)

    # 添加指标历史记录字典
    metrics_history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'dice': []
    }

    # 早停机制
    early_stopping = EarlyStopping(
        patience=config['train']['early_stopping']['patience'],
        min_delta=config['train']['early_stopping']['min_delta']
    )
    
    # 训练循环
    best_f1 = 0.0
    for epoch in range(config['train']['num_epochs']):
        model.train()
        epoch_loss = 0
        metrics_sum = {}
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["train"]["num_epochs"]}') as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)
                
                # 调整掩码维度
                masks = masks.squeeze(1)
                
                # 前向传播
                outputs, attention_maps = model(images)
                
                # 计算损失
                loss = criterion(outputs, masks)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 计算指标
                with torch.no_grad():
                    predictions = torch.sigmoid(outputs)
                    batch_metrics = evaluate_batch(
                        predictions.cpu().numpy(),
                        masks.cpu().numpy()
                    )


                # 更新进度条
                epoch_loss += loss.item()
                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': epoch_loss / (batch_idx + 1)
                })

                # 可视化
                if batch_idx % config['visualization']['frequency'] == 0:
                    visualizer.visualize_batch(
                        images.cpu(),
                        masks.cpu(),
                        predictions.cpu(),
                        attention_maps.detach().cpu(),
                        batch_idx,
                        epoch
                    )
        
        # 计算平均指标
        avg_loss = epoch_loss / len(train_loader)
        metrics_avg = {k: v / len(train_loader) for k, v in metrics_sum.items()}

        # 更新指标历史
        metrics_history['loss'].append(avg_loss)
        for metric_name, value in metrics_avg.items():
            metrics_history[metric_name].append(value)

        # 更新学习率（基于F1分数）
        current_f1 = metrics_avg.get('f1', 0)
        scheduler.step(current_f1)
        
        # 早停检查
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break
        
        # 保存最佳模型（基于F1分数）
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # 定期保存检查点
        if (epoch + 1) % config['train']['save_frequency'] == 0:
            save_training_process(
                save_dir,
                epoch,
                model,
                optimizer,
                avg_loss,
                metrics_avg
            )
        
        # 记录训练信息
        logging.info(f'Epoch {epoch+1}/{config["train"]["num_epochs"]}:')
        logging.info(f'Average Loss: {avg_loss:.4f}')
        for metric_name, value in metrics_avg.items():
            logging.info(f'{metric_name}: {value:.4f}')
    # 在训练结束后生成所有指标的可视化图表
    metrics_visualizer.visualize_all_metrics(metrics_history)
    # 保存最终的指标历史到文件
    save_training_process(
        save_dir,
        epoch,
        model,
        optimizer,
        avg_loss,
        {
            **metrics_avg,
            'metrics_history': metrics_history
        }
    )


if __name__ == '__main__':
    # 加载配置
    try:
        with open('configs/config.yaml', 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open('configs/config.yaml', 'r', encoding='gbk') as f:
            config = yaml.safe_load(f)
    
    train_model(config)