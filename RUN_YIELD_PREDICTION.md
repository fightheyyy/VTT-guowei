# 🚀 产量预测实验 - 快速运行指南

## 实验目标

**找到最短的有效预测天数**：测试不同输入长度（30-360天）对产量预测准确度的影响

## 一键运行

### 方式1：快速测试（推荐新手）⭐

```bash
python experiments/yield_prediction/train.py --quick
```

**特点**：
- 测试4个时间点：60天、120天、180天、300天
- 每个模型训练30轮
- 预计耗时：**1-2小时**（取决于GPU）
- 快速了解结果趋势

### 方式2：完整实验（推荐完整分析）

```bash
python experiments/yield_prediction/train.py
```

**特点**：
- 测试12个时间点：30、60、90、120、...、360天
- 每个模型训练50轮
- 预计耗时：**4-8小时**
- 详细分析最优时间窗口

### 方式3：自定义训练轮数

```bash
python experiments/yield_prediction/train.py --epochs 100
```

## 实验流程

运行后会自动执行：

```
1. 加载数据 (data/extract*.csv)
   ├── 训练集: 2019-2021年数据
   └── 测试集: 2022年数据

2. 对每个输入长度训练模型
   ├── 创建 LanguageOnlyYieldPredictor 模型
   ├── 训练 N 轮（快速=30轮，完整=50轮）
   ├── 保存最佳模型
   └── 记录性能指标

3. 生成结果
   ├── 结果JSON (results.json)
   ├── 性能图表 (analysis.png)
   └── TensorBoard日志
```

## 输出结果

### 1. 控制台输出

```
======================================================================
实验1: 产量预测 - 找到最短有效预测天数
======================================================================

[1/4] 训练产量预测模型 (输入长度: 6步 / 60天)
----------------------------------------------------------------------
  训练集: 1500 样本, 6步(60天)
  测试集: 500 样本, 6步(60天)
Epoch 1/30: 100%|████████████| train_loss=0.1234 test_loss=0.1567
Epoch 2/30: 100%|████████████| train_loss=0.0987 test_loss=0.1234
...
  ✓ 最佳模型: Epoch 15, Loss=0.0823
  
测试集性能:
  RMSE: 0.456 (原始单位)
  MAE:  0.342
  R²:   0.823
  MAPE: 8.5%

[2/4] 训练产量预测模型 (输入长度: 12步 / 120天)
...
```

### 2. 结果文件

#### `experiments/yield_prediction/results/results.json`
```json
{
  "6": {
    "rmse": 0.456,
    "mae": 0.342,
    "r2": 0.823,
    "mape": 8.5
  },
  "12": {
    "rmse": 0.389,
    "mae": 0.298,
    "r2": 0.876,
    "mape": 7.2
  },
  ...
}
```

#### `experiments/yield_prediction/results/analysis.png`
性能对比图表：
- 左图：RMSE vs 输入天数（标出最优点⭐）
- 右图：R² vs 输入天数

### 3. 保存的模型

```
experiments/yield_prediction/checkpoints/
├── model_6steps_best.pth      # 60天输入的最佳模型
├── model_12steps_best.pth     # 120天输入的最佳模型
├── model_18steps_best.pth     # 180天输入的最佳模型
└── model_30steps_best.pth     # 300天输入的最佳模型
```

### 4. 训练日志

```bash
# 查看TensorBoard
tensorboard --logdir=experiments/yield_prediction/logs

# 浏览器打开
http://localhost:6006
```

可查看：
- 训练/测试损失曲线
- 学习率变化
- 各输入长度的对比

## 数据说明

### 输入数据

- **来源**: `data/extract*.csv`
- **波段**: 7个主要遥感指标（NIR, RVI, SWIR1, blue, evi, ndvi, red）
- **时间步**: 36步（每步=10天，共360天=1年）
- **样本数**: 
  - 训练集: 1500样本（2019-2021）
  - 测试集: 500样本（2022）

### 输入长度对应关系

| 输入步数 | 天数 | 月数 | 说明 |
|---------|------|------|------|
| 3步 | 30天 | 1个月 | 极早期预测 |
| 6步 | 60天 | 2个月 | 早期预测 |
| 12步 | 120天 | 4个月 | 中早期预测 |
| 18步 | 180天 | 6个月 | 中期预测 |
| 24步 | 240天 | 8个月 | 中晚期预测 |
| 30步 | 300天 | 10个月 | 晚期预测 |
| 36步 | 360天 | 12个月 | 全年数据 |

## 模型说明

### LanguageOnlyYieldPredictor

```
输入: 时间序列 [Batch, input_steps, 7变量]
  ↓
Patchify & Embed (分patch + 嵌入)
  ↓
Transformer Encoder (语言模块)
  ↓
Variate Selection (变量选择)
  ↓
Regression Head (回归层)
  ↓
输出: 产量 [Batch, 1]
```

**特点**：
- 端到端训练（直接优化产量预测）
- 只用语言模态（比双模态效果更好）
- 无序列补全（避免误差传播）

## 评估指标

- **RMSE** (均方根误差): 越小越好，反映整体误差
- **MAE** (平均绝对误差): 越小越好，更直观
- **R²** (决定系数): 0-1之间，越接近1越好
- **MAPE** (平均绝对百分比误差): 百分比形式，越小越好

## 预期结果

根据之前的研究，预期：

1. **最短有效天数**: 可能在60-120天之间
2. **性能趋势**: 输入越长，预测越准确，但增益递减
3. **最优平衡点**: 预测准确度 vs 数据收集时间的平衡

## 常见问题

### Q1: GPU内存不足？

```bash
# 减小batch_size（修改data_loader.py）
# 或使用CPU
CUDA_VISIBLE_DEVICES="" python experiments/yield_prediction/train.py --quick
```

### Q2: 网络超时？

脚本已设置离线模式，模型会从缓存加载：
```python
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### Q3: 想更改波段？

修改 `experiments/yield_prediction/train.py`:
```python
selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
# 改成你想要的波段
```

### Q4: 想测试更多时间点？

修改 `experiments/yield_prediction/train.py`:
```python
# 完整模式
input_steps_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
# 改成任意步数列表
```

### Q5: 如何使用训练好的模型？

```python
import torch
from models.simple_yield_predictor import LanguageOnlyYieldPredictor

# 加载模型
model = LanguageOnlyYieldPredictor(
    input_time_steps=12,  # 对应120天
    n_variates=7
)
model.load_state_dict(torch.load('experiments/yield_prediction/checkpoints/model_12steps_best.pth'))
model.eval()

# 预测
with torch.no_grad():
    x = torch.randn(1, 12, 7)  # 你的数据
    yield_pred = model(x)
    # 记得反归一化: yield_pred * yield_std + yield_mean
```

## 下一步

实验完成后：

1. **查看结果**：
   ```bash
   # 查看JSON
   cat experiments/yield_prediction/results/results.json
   
   # 查看图表
   start experiments/yield_prediction/results/analysis.png  # Windows
   ```

2. **分析最优时间点**：
   - 找到RMSE最小的输入长度
   - 评估准确度vs时间成本的权衡

3. **进一步实验**：
   - 实验2：对比序列补全方法
   - 实验3：测试可变长度补全
   - 实验4：两阶段训练方法

## 技术细节

### 训练参数

- **优化器**: AdamW (lr=1e-4, weight_decay=1e-5)
- **损失函数**: MSELoss
- **学习率调度**: ReduceLROnPlateau (patience=5, factor=0.5)
- **早停**: patience=15轮
- **Batch Size**: 32

### 数据处理

- **归一化**: 序列和产量分别标准化（零均值，单位方差）
- **数据增强**: 无（使用原始数据）
- **验证方式**: 年份划分（2019-2021训练，2022测试）

## 引用

如果使用此代码，请引用：

```bibtex
@software{vtt2024,
  title={VTT: Variable-length Timeseries Transformer for Crop Yield Prediction},
  year={2024},
}
```

---

**立即开始**: `python experiments/yield_prediction/train.py --quick` 🚀

