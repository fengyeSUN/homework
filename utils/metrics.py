# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_f1_measure(output_image, gt_image, thre=0.1):
    """
    参考demo_MDvsFA_pytorch.py中的F1计算方法

    Args:
        output_image: 预测结果
        gt_image: 真实标签
        thre: 二值化阈值

    Returns:
        F1: F1分数
    """
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image > thre
    gt_bin = gt_image > thre
    recall = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
    prec = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
    F1 = 2 * recall * prec / np.maximum(0.001, recall + prec)
    return F1, prec, recall


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


def calculate_dice_coefficient(pred, target, threshold=0.2):
    """
    计算Dice系数
    Args:
        pred: 预测结果 (numpy array)
        target: 真实标签 (numpy array)
        threshold: 二值化阈值
    """
    # 将预测结果二值化
    pred = np.array(pred > threshold, dtype=np.float32)
    target = np.array(target > threshold, dtype=np.float32)

    # 计算真阳性、假阳性和假阴性
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    # 计算Dice系数
    # Dice = 2|X∩Y|/(|X|+|Y|) = 2*TP/(2*TP + FP + FN)
    numerator = 2.0 * tp
    denominator = 2.0 * tp + fp + fn
    dice = numerator / (denominator + 1e-6)  # 添加小量避免除零

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


def evaluate_batch(predictions, targets, threshold= 0.2):
    """
    计算一个批次的评估指标
    Args:
        predictions: 模型预测值 (numpy array)
        targets: 真实标签 (numpy array)
    """
    pred_binary = (predictions > threshold).astype(np.float32)
    targets_binary = targets.astype(np.float32)

    # 计算混淆矩阵元素
    tp = np.sum((pred_binary == 1) & (targets_binary == 1))
    fp = np.sum((pred_binary == 1) & (targets_binary == 0))
    fn = np.sum((pred_binary == 0) & (targets_binary == 1))
    tn = np.sum((pred_binary == 0) & (targets_binary == 0))

    # 计算准确率
    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)

    f1, precision, recall = calculate_f1_measure(predictions, targets, threshold)

    # IoU 计算
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + epsilon)

    # Dice 系数
    dice = calculate_dice_coefficient(predictions, targets, threshold)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }


def evaluate_model_metrics(predictions, targets, threshold=0.2):
    """
    计算整体模型评估指标
    Args:
        predictions: 所有预测值列表
        targets: 所有真实标签列表
        threshold: 二值化阈值
    """

    # 二值化预测结果用于其他指标计算
    pred_binary = (predictions > threshold).astype(np.int32)
    targets_binary = targets.astype(np.int32)

    # 计算准确率
    accuracy = accuracy_score(targets_binary.flatten(), pred_binary.flatten())

    f1, precision, recall = calculate_f1_measure(predictions, targets, threshold)

    # 计算IoU和Dice
    tp = np.sum((pred_binary == 1) & (targets_binary == 1))
    fp = np.sum((pred_binary == 1) & (targets_binary == 0))
    fn = np.sum((pred_binary == 0) & (targets_binary == 1))

    epsilon = 1e-7
    intersection = tp
    union = tp + fp + fn

    iou = intersection / (union + epsilon)
    dice = calculate_dice_coefficient(predictions, targets, threshold)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }