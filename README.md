# TimesCLIP - 双模态时间序列预测模型

基于论文思路实现的TimesCLIP模型核心架构，通过视觉和语言双模态融合进行时间序列预测。

## 项目结构

```
VTT/
├── models/
│   ├── __init__.py              # 模块导出
│   ├── preprocessor.py          # 数据预处理器
│   ├── vision_module.py         # 视觉模块
│   ├── language_module.py       # 语言模块
│   ├── alignment.py             # 多模态对齐
│   ├── variate_selection.py     # 变量选择模块
│   ├── generator.py             # 生成器
│   └── timesclip.py            # 完整模型
├── requirements.txt             # 依赖包
└── README.md                   # 项目说明
```

## 核心组件

1. **数据预处理器** - 将数值时间序列转换为视觉和语言两种模态
2. **视觉模块** - 使用预训练CLIP ViT提取图像特征（冻结）
3. **语言模块** - 使用CLIP Text Encoder处理数值patch序列（微调）
4. **多模态对齐** - 通过InfoNCE对比学习损失对齐双模态特征
5. **变量选择** - 使用交叉注意力识别关键变量
6. **生成器** - 基于融合特征输出最终预测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用示例

```python
import torch
from models import TimesCLIP

# 初始化模型
model = TimesCLIP(
    time_steps=96,           # 输入序列长度
    n_variates=7,            # 变量数量
    prediction_steps=96,     # 预测步长
    patch_length=16,         # patch长度
    stride=8,                # patch步长
    d_model=512,             # 隐藏维度
    n_heads=8,               # 注意力头数
    temperature=0.07         # 对比学习温度
)

# 输入数据
x = torch.randn(32, 96, 7)  # [Batch, Time_Steps, N_Variates]

# 前向传播
y_pred, contrastive_loss = model(x, return_loss=True)
# y_pred: [Batch, N_Variates, Prediction_Steps]

# 计算总损失
y_true = torch.randn(32, 7, 96)  # 真实值
total_loss, loss_dict = model.compute_loss(
    y_pred, y_true, contrastive_loss,
    lambda_gen=1.0,
    lambda_align=0.1
)

# 获取优化器参数组（差异化学习率）
param_groups = model.get_parameter_groups(
    lr_vision=1e-5,  # 视觉模块投影层
    lr_other=1e-4    # 其他部分
)
optimizer = torch.optim.AdamW(param_groups)
```

## 关键技术特性

- **实例归一化**: 在Time_Steps维度对每个变量独立归一化
- **预训练模型**: 使用CLIP ViT-B/16作为视觉骨干网络（冻结）
- **Transformer编码器**: 使用标准Transformer处理数值patch序列（可训练）
- **对比学习**: 通过InfoNCE损失对齐双模态特征
- **差异化学习率**: 视觉投影层和其他部分使用不同学习率

## 模型参数说明

- `time_steps`: 输入时间序列长度
- `n_variates`: 多变量时间序列的变量数量
- `prediction_steps`: 预测的未来步数
- `patch_length`: 将时间序列切分的patch长度
- `stride`: patch切分的步长
- `d_model`: 模型的隐藏维度
- `n_heads`: 多头注意力的头数
- `temperature`: InfoNCE损失的温度参数
- `image_size`: 时间序列可视化后的图像尺寸
- `clip_model_name`: 使用的CLIP预训练模型

## 训练建议

1. 使用AdamW优化器
2. 为预训练部分设置较小学习率（1e-5）
3. 为新增部分设置较大学习率（1e-4）
4. 生成损失权重λ₁=1.0，对齐损失权重λ₂=0.1
5. 首次运行会自动下载CLIP预训练模型

## 注意事项

- 首次运行需要下载CLIP预训练模型（约1GB）
- 图像生成使用matplotlib，需要足够内存
- 建议使用GPU训练，模型包含大量Transformer层

