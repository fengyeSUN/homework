import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedAttentionModule(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()

        # 简化的亮点增强模块
        self.brightness_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )

        # 保留原有的多尺度特征提取(使用更适合的尺度)
        self.scales = [1, 2]  # 减少尺度数量,专注于小目标
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, 3, padding=s, dilation=s)
            for s in self.scales
        ])

        # 改进的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )

        # 优化的空间注意力,使用较小的卷积核
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),  # 减小卷积核
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Conv2d(hidden_dim * len(self.scales), in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # 亮点增强
        bright_attention = self.brightness_attention(x)
        x = x * bright_attention

        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.convs:
            multi_scale_features.append(conv(x))

        features = torch.cat(multi_scale_features, dim=1)
        features = self.fusion(features)
        features = self.norm(features)

        # 通道注意力
        channel_weights = self.channel_attention(features)
        features = features * channel_weights

        # 空间注意力
        avg_out = torch.mean(features, dim=1, keepdim=True)
        max_out, _ = torch.max(features, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial)

        # 残差连接
        out = features * spatial_weights + x

        return out, spatial_weights


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = ImprovedAttentionModule(in_channels, out_channels)

    def forward(self, x):
        return self.attention(x)