# TimesCLIP 训练指南

## 数据说明

### 数据集结构
- **文件**: `extract2022_20251010_165007.csv`
- **数据内容**: 2022年多波段遥感时间序列数据
- **时间步**: 每个波段36个时间点（全年，步长10天）
- **波段列表**: NIR, RVI, SWIR1, blue, bsi, evi, gcvi, green, lswi, ndsi, ndvi, ndwi, ndyi, red

### 数据格式
每一行代表一个样本，每个波段有36列：
- `NIR_00, NIR_01, ..., NIR_35` (近红外波段)
- `RVI_00, RVI_01, ..., RVI_35` (比值植被指数)
- `SWIR1_00, SWIR1_01, ..., SWIR1_35` (短波红外1)
- ... 其他波段类似

**注**: y2019, y2020, y2021列暂时忽略，只使用2022年数据

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

需要的包：
- torch >= 2.0.0
- torchvision
- transformers
- matplotlib
- pandas
- scikit-learn
- tqdm
- tensorboard

### 2. 测试数据加载器
```bash
python data_loader.py
```

这将验证数据是否正确加载，并显示数据形状。

### 3. 快速测试模型（可选）
```bash
python quick_test.py
```

验证模型结构是否正常工作。

### 4. 开始训练

#### 方式1：使用默认配置
```bash
python train.py
```

#### 方式2：自定义配置
编辑 `train.py` 中的配置参数：

```python
train(
    csv_path="extract2022_20251010_165007.csv",
    
    # 选择波段（None表示使用全部14个波段）
    selected_bands=['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red'],
    
    # 时间序列配置
    lookback=24,           # 输入：前24个时间步
    prediction_steps=12,   # 预测：接下来12个时间步
    
    # 训练配置
    batch_size=16,
    epochs=50,
    
    # 学习率
    lr_vision=1e-5,        # 视觉模块学习率
    lr_other=1e-4,         # 其他模块学习率
    
    # 损失权重
    lambda_gen=1.0,        # MSE损失权重
    lambda_align=0.1,      # 对比学习损失权重
    
    # 模型配置
    d_model=256,           # 隐藏维度（256较快，512更准确）
    patch_length=8,        # patch长度
    stride=4,              # patch步长
    
    # 设备
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

## 训练过程说明

### 数据划分
- 训练集: 80%
- 测试集: 20%
- 随机种子: 42（保证可复现）

### 训练流程
1. **数据加载**: 从CSV读取多波段数据
2. **数据分割**: 
   - 输入 X: 前`lookback`个时间步 → `[batch, lookback, n_variates]`
   - 目标 Y: 接下来`prediction_steps`个时间步 → `[batch, n_variates, prediction_steps]`
3. **双模态处理**:
   - 视觉分支: 将时间序列绘制为彩色折线图
   - 语言分支: 将时间序列分块处理
4. **模型训练**:
   - 对比学习: 对齐视觉和语言特征
   - 预测任务: MSE损失
   - 总损失 = λ₁ × MSE + λ₂ × Contrastive

### 监控训练

#### 使用TensorBoard
```bash
tensorboard --logdir=logs
```

然后在浏览器打开 `http://localhost:6006`

可以查看：
- 训练/验证损失曲线
- MSE损失
- 对比学习损失
- 学习率变化

#### 查看保存的模型
训练过程中会自动保存：
- `checkpoints/best_model.pth` - 验证集最佳模型
- `checkpoints/checkpoint_epoch_*.pth` - 每10个epoch的检查点

## 推理和评估

### 1. 加载模型并评估
```bash
python inference.py
```

这将：
- 加载最佳模型
- 在测试集上评估
- 计算MSE、MAE、RMSE指标
- 可视化前5个样本的预测结果（保存到`predictions/`目录）

### 2. 自定义推理
```python
from inference import load_model, predict

# 加载模型
model, config = load_model('checkpoints/best_model.pth')

# 准备输入数据 [lookback, n_variates]
x = ...  # 你的输入数据

# 预测
y_pred = predict(model, x)  # [n_variates, prediction_steps]
```

## 配置建议

### 快速实验（调试用）
```python
lookback=16
prediction_steps=8
batch_size=32
d_model=128
epochs=10
```

### 标准训练
```python
lookback=24
prediction_steps=12
batch_size=16
d_model=256
epochs=50
```

### 高精度训练
```python
lookback=28
prediction_steps=8
batch_size=8
d_model=512
epochs=100
```

## 波段选择建议

### 全波段（14个）
```python
selected_bands=None  # 使用所有波段
```

### 常用植被指数（7个）
```python
selected_bands=['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
```

### 核心植被指数（4个）
```python
selected_bands=['NIR', 'evi', 'ndvi', 'red']
```

## 常见问题

### 1. 内存不足
- 减小 `batch_size`
- 减小 `d_model`
- 使用更少的波段

### 2. 训练速度慢
- 确保使用GPU（`device='cuda'`）
- 减小 `d_model`
- 减小 `lookback`

### 3. 过拟合
- 增加数据量
- 减小 `d_model`
- 增加 `lambda_align`（对比学习权重）

### 4. 欠拟合
- 增大 `d_model`
- 增加 `epochs`
- 调整学习率

## 输出文件结构

```
VTT/
├── checkpoints/              # 模型检查点
│   ├── best_model.pth       # 最佳模型
│   └── checkpoint_epoch_*.pth
├── logs/                     # TensorBoard日志
├── predictions/              # 预测可视化结果
│   ├── sample_1.png
│   └── ...
└── extract2022_20251010_165007.csv  # 原始数据
```

## 预期结果

训练50个epoch后，在测试集上应该能达到：
- MSE: < 500000（取决于数据尺度）
- 对比学习损失: < 2.0
- 视觉和语言特征能成功对齐

预测曲线应该能较好地跟随真实趋势。

## 下一步

训练完成后，你可以：
1. 调整超参数优化性能
2. 尝试不同的波段组合
3. 调整输入和预测的时间窗口
4. 使用模型进行实际预测任务

