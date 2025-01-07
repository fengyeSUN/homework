# -*- coding: utf-8 -*-
import yaml
import os

# 确保configs目录存在
os.makedirs('configs', exist_ok=True)

config = {
    'model': {
        'num_classes': 1,  # 修改为1，因为是二分类问题
        'hidden_dim': 128,
        'num_attention_layers': 2,
        'backbone': 'resnet18'
    },
    'train': {
        'batch_size': 8,          # 增大批次大小
        'num_epochs': 100,         # 增加训练轮数
        'learning_rate': 5e-5,     # 降低学习率
        'weight_decay': 1e-5,
        'save_dir': './train_model',
        'save_frequency': 5,
        'early_stopping': {
            'patience': 15,
            'min_delta': 0.0005
        },
        'loss_weights': {          # 损失函数的权重
            'focal': 0.5,
            'dice': 0.3,
            'bce': 0.2
        },
        'focal_loss': {            # Focal Loss参数
            'alpha': 0.75,
            'gamma': 2
        },
        'pos_weight': 30.0   #原20.0      # BCE正样本权重
    },
    'data': {
        'train_dir': './train',
        'test_dir': './test',
        'image_size': 256,
        'num_workers': 4,
        'augmentation': {
            'rotation_range': 20,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': 0.15,
            'contrast_range': 0.15
        }
    },
    'visualization': {
        'frequency': 100,
        'num_visualizations': 8,
        'save_attention_maps': True,
        'save_predictions': True
    }
}

# 保存配置文件
with open('config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print("配置文件已创建：configs/config.yaml")