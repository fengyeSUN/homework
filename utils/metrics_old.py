# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_f1_score(pred, target, threshold=0.2):
    """
    计算F1分数、精确率和召回率
    Args:
        pred: 预测结果 (numpy array)
        target: 真实标签 (numpy array)
        threshold: 二值化阈值

    Returns:
        precision: 精确率
        recall: 召回率
        f1: F1分数
    """
    # 将预测结果二值化
    pred = np.array(pred > threshold, dtype=np.float32)
    target = np.array(target > threshold, dtype=np.float32)

    # 计算TP, FP, FN
    intersection = (pred * target).sum()
    precision = intersection / (pred.sum() + 1e-6)  # 避免除零
    recall = intersection / (target.sum() + 1e-6)

    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


def calculate_iou(pred, target, threshold=0.3):
    """
    计算IoU (Intersection over Union)

    Args:
        pred: 预测结果 (numpy array)
        target: 真实标签 (numpy array)
        threshold: 二值化阈值

    Returns:
        iou: IoU分数
    """
    # 将预测结果二值化
    pred = np.array(pred > threshold, dtype=np.float32)
    target = np.array(target > threshold, dtype=np.float32)

    # 计算交集和并集
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection  # 减去重复计算的交集

    # 计算IoU
    iou = intersection / (union + 1e-6)  # 避免除零

    return iou


def calculate_dice_coefficient(pred, target, threshold=0.5):
    """
    计算Dice系数

    Args:
        pred: 预测结果 (numpy array)
        target: 真实标签 (numpy array)
        threshold: 二值化阈值

    Returns:
        dice: Dice系数
    """
    # 将预测结果二值化
    pred = np.array(pred > threshold, dtype=np.float32)
    target = np.array(target > threshold, dtype=np.float32)

    # 计算Dice系数
    intersection = 2.0 * (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = intersection / (union + 1e-6)

    return dice


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """
    计算像素级准确率

    Args:
        pred: 预测结果 (numpy array)
        target: 真实标签 (numpy array)
        threshold: 二值化阈值

    Returns:
        accuracy: 像素级准确率
    """
    # 将预测结果二值化
    pred = np.array(pred > threshold, dtype=np.float32)
    target = np.array(target > threshold, dtype=np.float32)

    # 计算正确预测的像素数
    correct = (pred == target).sum()
    total = pred.size

    accuracy = correct / total

    return accuracy

def evaluate_batch(predictions, targets, threshold=0.25):  # 降低评估阈值
#def evaluate_batch(predictions, targets, threshold=0.5):
    """
    计算一个批次的评估指标
    Args:
        predictions: 模型预测值 (numpy array)
        targets: 真实标签 (numpy array)
        threshold: 二值化阈值
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 确保输入是numpy数组
    predictions = np.array(predictions)
    targets = np.array(targets)

    # 确保维度匹配
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    # 二值化预测结果
    pred_binary = (predictions > threshold).astype(np.float32)
    targets_binary = targets.astype(np.float32)

    # 计算混淆矩阵元素
    tp = np.sum((pred_binary == 1) & (targets_binary == 1))
    fp = np.sum((pred_binary == 1) & (targets_binary == 0))
    fn = np.sum((pred_binary == 0) & (targets_binary == 1))
    tn = np.sum((pred_binary == 0) & (targets_binary == 0))

    # 计算指标
    epsilon = 1e-7  # 防止除零
    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # F1 分数计算
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    # IoU 计算
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + epsilon)

    # Dice 系数
    dice = 2 * intersection / (2 * intersection + fp + fn + epsilon)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }
def evaluate_model_metrics(predictions, targets, threshold=0.3):
    """
    计算整体模型评估指标
    Args:
        predictions: 所有预测值列表
        targets: 所有真实标签列表
        threshold: 二值化阈值
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 确保输入是numpy数组并展平
    predictions = np.array(predictions).reshape(-1)
    targets = np.array(targets).reshape(-1)
    
    # 二值化预测结果
    pred_binary = (predictions > threshold).astype(np.int32)
    targets_binary = targets.astype(np.int32)
    
    # 使用sklearn计算指标
    metrics = {
        'accuracy': accuracy_score(targets_binary, pred_binary),
        'precision': precision_score(targets_binary, pred_binary, zero_division=0),
        'recall': recall_score(targets_binary, pred_binary, zero_division=0),
        'f1': f1_score(targets_binary, pred_binary, zero_division=0)
    }
    
    # 计算IoU和Dice
    tp = np.sum((pred_binary == 1) & (targets_binary == 1))
    fp = np.sum((pred_binary == 1) & (targets_binary == 0))
    fn = np.sum((pred_binary == 0) & (targets_binary == 1))
    
    epsilon = 1e-7
    intersection = tp
    union = tp + fp + fn
    
    metrics['iou'] = intersection / (union + epsilon)
    metrics['dice'] = 2 * intersection / (2 * intersection + fp + fn + epsilon)
    
    return metrics