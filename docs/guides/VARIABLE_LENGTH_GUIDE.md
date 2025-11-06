# 可变长度预测功能说明

## 功能介绍

这个功能允许你**使用任意前N个月的数据预测剩余的月份**，实现灵活的早期预测。

### 核心优势

1. **灵活性**：输入长度可以是3-30个月中的任意值
2. **早期预测**：数据越早获得，决策价值越高
3. **一个模型**：训练一次，支持所有输入长度
4. **实用性**：适应真实场景中数据可能不完整的情况

## 应用场景

| 输入月数 | 预测月数 | 应用场景 | 价值 |
|---------|---------|---------|------|
| 3个月 | 33个月 | 极早期预估 | 市场规划、期货交易 |
| 6个月 | 30个月 | 早期预警 | 调整种植策略 |
| 12个月 | 24个月 | 中期预测 | 优化资源配置 |
| 18个月 | 18个月 | 准确预测 | 制定收获计划 |
| 24个月 | 12个月 | 后期确认 | 精准产量估算 |

## 架构设计

### 模型结构

```
输入: [Batch, Input_Length, N_Variates]
         ↓
    LanguageModule (BERT编码)
         ↓
    变量选择 + 特征融合
         ↓
    Transformer Decoder (自回归生成)
         ↓
输出: [Batch, Prediction_Length, N_Variates]
```

### 关键特性

1. **动态Padding**：处理不同长度的输入
2. **位置编码**：保留时间序列的位置信息
3. **长度嵌入**：让模型知道当前输入长度
4. **Decoder架构**：基于已知序列生成未来序列

## 使用方法

### 1. 训练模型

```bash
python train_variable_length.py
```

训练参数：
- `max_time_steps=36`：完整序列长度
- `min_input_length=3`：最短输入3个月
- `max_input_length=30`：最长输入30个月
- 训练时会随机采样不同输入长度，让模型适应所有情况

### 2. 测试不同长度

```bash
python test_variable_length.py
```

会测试3、6、12、18、24、30个月输入的预测效果。

### 3. 实际使用

```python
import torch
from models.timesclip_variable_length import TimesCLIPVariableLength

# 加载模型
model = TimesCLIPVariableLength(
    max_time_steps=36,
    n_variates=7,
    patch_length=6,
    stride=3,
    d_model=256
)
model.load_state_dict(torch.load('checkpoints/variable_length_best.pth'))
model.eval()

# 场景1：只有前3个月数据
x_3months = torch.randn(1, 3, 7)  # [Batch=1, Time=3, Vars=7]
pred_33months = model(x_3months, input_length=3)
print(f"预测未来33个月: {pred_33months.shape}")  # [1, 33, 7]

# 场景2：有前12个月数据
x_12months = torch.randn(1, 12, 7)
pred_24months = model(x_12months, input_length=12)
print(f"预测未来24个月: {pred_24months.shape}")  # [1, 24, 7]

# 场景3：预测完整序列（包含已知部分）
full_sequence = model.predict_full_sequence(x_12months)
print(f"完整36个月序列: {full_sequence.shape}")  # [1, 36, 7]
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `models/timesclip_variable_length.py` | 可变长度模型定义 |
| `data_loader_variable_length.py` | 可变长度数据加载器 |
| `train_variable_length.py` | 训练脚本 |
| `test_variable_length.py` | 测试脚本 |

## 训练策略

### 随机长度训练

在训练时，每个batch中的样本会随机选择输入长度（3-30个月），这样模型学会：

```python
# Epoch 1, Batch 1
样本1: 前15个月 → 预测后21个月
样本2: 前8个月  → 预测后28个月
样本3: 前22个月 → 预测后14个月
样本4: 前10个月 → 预测后26个月

# Epoch 1, Batch 2
样本1: 前5个月  → 预测后31个月
样本2: 前18个月 → 预测后18个月
...
```

这样训练出的模型能适应所有输入长度。

### 测试固定长度

测试时固定输入长度，评估模型在该长度下的性能：

```python
# 测试前18个月的预测能力
test_loader = create_dataloaders(..., test_input_length=18)
```

## 性能预期

根据输入长度不同，预测准确度会有差异：

| 输入长度 | 预期RMSE | 预测难度 | 应用价值 |
|---------|---------|---------|---------|
| 3个月  | 较高 | 极高 | ⭐⭐⭐⭐⭐ |
| 6个月  | 高 | 高 | ⭐⭐⭐⭐ |
| 12个月 | 中 | 中 | ⭐⭐⭐ |
| 18个月 | 低 | 低 | ⭐⭐ |
| 24个月 | 很低 | 很低 | ⭐ |

**关键洞察**：虽然输入越少准确度越低，但预测价值反而越高！

## 与现有模型对比

| 特性 | 固定长度模型 | 可变长度模型 |
|------|------------|------------|
| 输入长度 | 固定18个月 | 任意3-30个月 |
| 灵活性 | ❌ | ✅ |
| 早期预测 | ❌ | ✅ |
| 训练复杂度 | 低 | 中 |
| 推理灵活度 | 低 | 高 |
| 实用价值 | 中 | 高 |

## 实际案例

### 案例1：农作物产量早期预警

**背景**：农业部门希望在播种后3个月就预测全年产量。

**方案**：
```python
# 3月份（播种后3个月）
x_march = load_data(months=['Jan', 'Feb', 'Mar'])  # [1, 3, 7]
pred_remaining = model(x_march, input_length=3)    # [1, 33, 7]

# 提取关键时间点预测
pred_harvest = pred_remaining[:, -6:, :]  # 收获期的6个月
```

**价值**：提前9个月预测产量，为市场调配提供充足时间。

### 案例2：灵活应对数据缺失

**背景**：由于天气原因，某些月份的卫星数据缺失。

**方案**：
```python
# 有数据的月份：1-8月
available_data = [1, 2, 3, 4, 5, 6, 7, 8]
x = load_data(months=available_data)  # [1, 8, 7]
pred = model(x, input_length=8)       # [1, 28, 7]

# 模型自动适应8个月输入，预测剩余28个月
```

## 未来扩展

### 1. 不连续输入
支持输入非连续的月份（如1-3月 + 6-8月）

### 2. 多尺度预测
同时输出月度、季度、年度预测

### 3. 不确定性估计
输出预测的置信区间

## 总结

可变长度预测功能提供了：

✅ **灵活性**：任意输入长度  
✅ **实用性**：适应真实场景  
✅ **早期预测**：最大化决策价值  
✅ **一致性**：单一模型支持所有场景  

这是对固定长度模型的重要补充，使模型更加实用和灵活！

