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
        
        # 验证目录存在
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        print(f"Looking for images in: {self.image_dir}")
        print(f"Looking for masks in: {self.mask_dir}")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 基础变换
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 训练时的数据增强
        # 增强训练数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=10
            ),
            # 添加随机裁剪和缩放
            transforms.RandomResizedCrop(
                size=256,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            )
        ])

        # 测试时的变换
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            self.normalize
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # 加载图像和掩码
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('L')

        # 应用变换
        if self.split == 'train':
            # 确保图像和掩码使用相同的随机变换
            seed = random.randint(0, 2**32)
            
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.train_transform(image)
            
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.train_transform(mask)

        # 应用基础变换
        image = self.test_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # 使用float而不是long

        return image, mask
