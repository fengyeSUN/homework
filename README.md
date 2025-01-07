# 小目标分割项目

基于视觉认知机理的注意力机制的小目标分割模型实现。

## 特点

- 多尺度注意力机制
- 通道注意力和空间注意力的结合
- 完整的训练和评估流程
- 详细的可视化支持

## 安装

```
pip install -r requirements.txt
```

## 使用方法

1. 准备数据

   ```
   data/
   ├── train/
   │ ├── image/
   │ └── mask/
   └── test/
   ├── image/
   └── mask/

2. 创建配置文件

   ```
   python create_config.py

3. 训练模型

   ```
   python train.py

4. 评估模型

   ```
   python evaluate.py
   ```

   ## 配置说明

   配置文件位于 `configs/config.yaml`，包含以下主要参数：

   - model: 模型相关配置
   - train: 训练相关配置
   - data: 数据相关配置
   - visualization: 可视化相关配置

   ## 结果可视化

   训练过程中的可视化结果保存在：

   ```
   train_model/
   ├── attention_maps/
   ├── predictions/
   └── metrics/
   ```

   ## 注意事项

   - 确保数据集格式正确
   - 根据显存大小调整batch_size
   - 适当调整学习率和训练轮数

5. 创建requirements.txt

   ```
   torch>=1.8.0
   torchvision>=0.9.0
   numpy>=1.19.2
   Pillow>=8.0.0
   matplotlib>=3.3.2
   tqdm>=4.50.2
   PyYAML>=5.3.1
   seaborn>=0.11.0
   scikit-learn>=0.23.2