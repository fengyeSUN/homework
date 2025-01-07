# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.scales = [1, 2, 4]  # 多尺度注意力
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, 3, padding=s, dilation=s)
            for s in self.scales
        ])
        
        self.attention_conv = nn.Conv2d(hidden_dim * len(self.scales), in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        features = []
        for conv in self.convs:
            features.append(conv(x))
        
        multi_scale_features = torch.cat(features, dim=1)
        attention = torch.sigmoid(self.attention_conv(multi_scale_features))
        
        out = x * attention
        out = self.norm(out)
        out = self.activation(out)
        
        return out, attention

class AttentionModule(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.multi_scale_attention = MultiScaleAttention(in_channels, hidden_dim)

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        x = x * spatial

        # 多尺度注意力
        x, attention_maps = self.multi_scale_attention(x)

        return x, attention_maps
