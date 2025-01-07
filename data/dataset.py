# -*- coding: utf-8 -*-
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np

class SmallObjectDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.image_dir = os.path.join(root_dir, 'image')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.split = split
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        # 针对灰度图优化的归一化参数
        self.normalize = transforms.Normalize(
            mean=[0.15],  # 调整为更适合灰度图的均值# 降低均值使亮点更突出
            std=[0.35]  # 调整为更适合灰度图的标准差
        )
        # 优化的训练数据增强
        self.train_transform = transforms.Compose([
            # 保持亮点特征的数据增强
            transforms.RandomRotation(5, fill=0),  # 减小旋转角度，黑色填充
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.02,  # 减小亮度变化，保持亮点特征
                contrast=0.02,  # 减小对比度变化范围
            ),
            # 局部增强
            transforms.RandomAffine(
                degrees=0,
                translate=(0.02, 0.02),  # 减小平移范围
                scale=(0.98, 1.02),  # 减小缩放范围
                shear=0  # 移除剪切，保持亮点形状
            ),
            # 保持目标完整性的裁剪
            transforms.RandomResizedCrop(
                size=256,
                scale=(0.99, 1.0),  # 接近原始大小
                ratio=(0.99, 1.01)  # 几乎保持原始比例
            )
        ])
        # 优化的测试变换
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            self.normalize
        ])
        # 优化的掩码变换
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        # 加载和预处理图像
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('L')
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('L')
        # 图像增强预处理
        if self.split == 'train':
            # 确保图像和掩码使用相同的随机种子
            seed = torch.randint(0, 2147483647, (1,)).item()
            torch.manual_seed(seed)
            random.seed(seed)
            # 应用变换
            image = self.train_transform(image)
            mask = self.train_transform(mask)
        else:
            image = self.test_transform(image)
            mask = self.mask_transform(mask)
        # 优化的掩码处理
        mask = (mask > 0.1).float()  # 降低阈值以捕获更多潜在目标
        # 转换为三通道（保持与原始代码兼容）
        image = torch.stack([image[0]] * 3, dim=0)
        return image, mask
