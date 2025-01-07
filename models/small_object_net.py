# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        # 反转通道列表，使其从高层到低层
        in_channels_list = in_channels_list[::-1]  # [512, 256, 128, 64]

        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # 反转特征列表，使其从高层到低层
        features = features[::-1]  # 从高层到低层

        last_inner = None
        output_features = []

        for idx, (feature, inner_block, layer_block) in enumerate(zip(
                features, self.inner_blocks, self.layer_blocks)):

            if last_inner is None:
                last_inner = inner_block(feature)
            else:
                # 上采样并相加
                inner_top_down = F.interpolate(last_inner,
                                               size=feature.shape[-2:],
                                               mode='nearest')
                last_inner = inner_block(feature) + inner_top_down

            output_features.append(layer_block(last_inner))

        # 反转输出特征列表，使其从低层到高层
        return output_features[::-1]

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.conv1(x)
        attention_weights = self.attention(features)
        refined_features = features * attention_weights
        output = self.conv2(refined_features)
        return output, attention_weights


class SmallObjectNet(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256):
        super().__init__()

        # Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 提取各个阶段的特征
        self.stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.stage2 = resnet.layer1  # 64 channels
        self.stage3 = resnet.layer2  # 128 channels
        self.stage4 = resnet.layer3  # 256 channels
        self.stage5 = resnet.layer4  # 512 channels

        # 特征金字塔
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=hidden_dim
        )

        # 注意力模块
        self.attention = AttentionModule(hidden_dim, hidden_dim)
        
        # 改进的解码器
        self.decoder = nn.ModuleList([
            # 1/32 -> 1/16
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2)
            ),
            # 1/16 -> 1/8
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2)
            ),
            # 1/8 -> 1/4
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 8),
                nn.ReLU(inplace=True)
            ),
            # 1/4 -> 1/2
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 16),
                nn.ReLU(inplace=True)
            ),
            # 1/2 -> 1/1
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim // 16, num_classes, kernel_size=4, stride=2, padding=1)
            )
        ])

        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化模型权重"""
        for m in self.decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入大小用于最终插值
        input_size = x.shape[-2:]

        # 提取多尺度特征
        x1 = self.stage1(x)  # 1/4, 64
        x2 = self.stage2(x1)  # 1/4, 64
        x3 = self.stage3(x2)  # 1/8, 128
        x4 = self.stage4(x3)  # 1/16, 256
        x5 = self.stage5(x4)  # 1/32, 512

        # FPN处理
        fpn_features = self.fpn([x2, x3, x4, x5])

        # 注意力处理
        x, attention_maps = self.attention(fpn_features[-1])

        # 解码
        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        # 确保输出大小与输入一致
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x.squeeze(1), attention_maps