# 灵活配置指南 - 不同输入月份的训练与预测

## 📊 核心概念

**是的！可以用更少的月份数据进行预测**，只需调整 `lookback` 参数。

- ✅ **优点**: 可以更早做出预测（如2月就能预测全年）
- ⚠️ **缺点**: 输入信息越少，预测精度会下降

---

## 📅 不同输入月份配置表

### 时间对应关系

全年36个时间步，每步约10天（36步 ≈ 360天 ≈ 12个月）

| 输入月份 | 时间步 (lookback) | 预测月份 | 预测步 (prediction_steps) | 难度 | 推荐精度 |
|---------|------------------|---------|-------------------------|------|---------|
| 1-2月 | 6 | 3-12月 | 30 | ⭐⭐⭐⭐⭐ | 低 |
| 1-3月 | 9 | 4-12月 | 27 | ⭐⭐⭐⭐ | 中低 |
| 1-4月 | 12 | 5-12月 | 24 | ⭐⭐⭐ | 中 |
| 1-5月 | 15 | 6-12月 | 21 | ⭐⭐⭐ | 中高 |
| 1-6月 | 18 | 7-12月 | 18 | ⭐⭐ | 高 |
| 1-7月 | 21 | 8-12月 | 15 | ⭐⭐ | 很高 |
| 1-8月 | 24 | 9-12月 | 12 | ⭐ | 很高 |

### 精度影响因素

1. **输入信息量**: 越多月份 → 信息越充分 → 精度越高
2. **季节性规律**: 如果作物生长有明显季节性，早期月份也能提供重要信息
3. **数据质量**: 高质量的观测数据能部分弥补输入长度不足

---

## 🚀 使用方法

### 方法1: 使用命令行参数

#### 训练（指定输入月份）

```bash
# 训练：使用1-2月数据预测3-12月
python train_flexible.py --input_months 2

# 训练：使用1-3月数据预测4-12月
python train_flexible.py --input_months 3

# 训练：使用1-6月数据预测7-12月
python train_flexible.py --input_months 6

# 完整参数示例
python train_flexible.py \
    --input_months 3 \
    --batch_size 16 \
    --epochs_stage1 50 \
    --epochs_stage2 100 \
    --d_model 256 \
    --device cuda
```

#### 预测

```bash
# 预测：使用训练时相同的月份数
python predict_flexible.py --input_months 2 --visualize

# 或
python predict_flexible.py --input_months 3 --visualize
```

### 方法2: 修改train_two_stage.py

手动指定lookback和prediction_steps：

```python
# 在train_two_stage.py中修改

# 示例1: 1-2月 → 3-12月
train_stage1_timeseries(
    lookback=6,              # 1-2月
    prediction_steps=30,     # 3-12月
    ...
)

# 示例2: 1-3月 → 4-12月
train_stage1_timeseries(
    lookback=9,              # 1-3月
    prediction_steps=27,     # 4-12月
    ...
)

# 示例3: 1-6月 → 7-12月
train_stage1_timeseries(
    lookback=18,             # 1-6月
    prediction_steps=18,     # 7-12月
    ...
)
```

---

## 📈 精度对比实验

### 建议的测试方案

训练多个不同配置的模型，对比验证集精度：

```bash
# 配置1: 最少输入（1-2月）
python train_flexible.py --input_months 2 --save_dir checkpoints/2months

# 配置2: 较少输入（1-3月）
python train_flexible.py --input_months 3 --save_dir checkpoints/3months

# 配置3: 中等输入（1-5月，推荐）
python train_flexible.py --input_months 5 --save_dir checkpoints/5months

# 配置4: 较多输入（1-6月）
python train_flexible.py --input_months 6 --save_dir checkpoints/6months
```

然后比较各模型在测试集上的表现。

---

## 💡 实际应用建议

### 场景1: 早期预测（2-3月）

**适用**: 需要尽早做出决策  
**配置**: `--input_months 2` 或 `--input_months 3`  
**注意**: 
- 精度会较低，建议作为参考
- 可以每个月重新预测，随着数据增多逐步提高精度

```bash
# 2月时的预测
python train_flexible.py --input_months 2
python predict_flexible.py --input_months 2

# 3月时更新预测
python train_flexible.py --input_months 3
python predict_flexible.py --input_months 3
```

### 场景2: 平衡预测（4-5月）

**适用**: 平衡时效性和精度  
**配置**: `--input_months 4` 或 `--input_months 5`  
**推荐**: ⭐⭐⭐ 这是较好的平衡点

```bash
python train_flexible.py --input_months 5
python predict_flexible.py --input_months 5
```

### 场景3: 高精度预测（6-7月）

**适用**: 追求高精度  
**配置**: `--input_months 6` 或更多  
**注意**: 预测时间范围变小，但精度更高

```bash
python train_flexible.py --input_months 6
python predict_flexible.py --input_months 6
```

---

## 🔧 高级技巧

### 1. 渐进式预测

随着时间推移，不断用新数据重新预测：

```python
# 2月：用1-2月数据
result_feb = predict_with_months(2)

# 3月：用1-3月数据（精度更高）
result_mar = predict_with_months(3)

# 4月：用1-4月数据（精度更高）
result_apr = predict_with_months(4)

# 观察预测如何随数据增多而稳定
```

### 2. 集成预测

训练多个不同输入长度的模型，然后集成：

```python
# 训练3个模型
model_3months = train(input_months=3)
model_5months = train(input_months=5)
model_6months = train(input_months=6)

# 集成预测（加权平均）
pred_final = 0.2 * pred_3 + 0.3 * pred_5 + 0.5 * pred_6
```

### 3. 不确定性估计

对于早期预测（输入少），可以提供置信区间：

```python
# 多次预测取均值和标准差
predictions = []
for _ in range(10):
    pred = model.predict(input_with_noise)
    predictions.append(pred)

mean_pred = np.mean(predictions)
std_pred = np.std(predictions)

print(f"预测: {mean_pred:.2f} ± {std_pred:.2f}")
```

---

## ⚠️ 注意事项

### 1. 训练数据匹配

- 训练时用什么输入长度，预测时就用相同长度
- 不能用训练6个月的模型去预测3个月的输入

### 2. 时间步计算

```python
# 确保: lookback + prediction_steps = 36
lookback = input_months * 3
prediction_steps = 36 - lookback
```

### 3. 最小输入限制

- 理论上最少可以用1个月（3步）
- 但太少会导致模型无法学习到有效模式
- **建议最少2-3个月**（6-9步）

### 4. 模型容量

输入越少，可以适当减小模型：

```python
# 输入少时
--d_model 128  # 而不是256

# 输入多时
--d_model 512  # 更大的模型
```

---

## 📊 预期精度对比

基于经验的估计（实际需要实验验证）：

| 输入月份 | 时间序列MSE | 产量预测MAE | 整体可用性 |
|---------|------------|------------|----------|
| 1-2月 | 很高 | 高 | ⚠️ 参考 |
| 1-3月 | 高 | 中高 | ✓ 可用 |
| 1-4月 | 中 | 中 | ✓ 较好 |
| 1-5月 | 中低 | 低 | ✓✓ 推荐 |
| 1-6月 | 低 | 很低 | ✓✓✓ 优秀 |

---

## 📝 完整示例

### 完整的2-3-5月渐进预测

```bash
# 步骤1: 2月初，用1-2月数据
echo "=== 2月预测 ==="
python train_flexible.py --input_months 2 \
    --save_dir checkpoints/feb \
    --epochs_stage1 30  # 可以少训练几轮

python predict_flexible.py --input_months 2 \
    --stage1_checkpoint checkpoints/feb/stage1_timeseries_best.pth \
    --stage2_checkpoint checkpoints/feb/stage2_yield_best.pth

# 步骤2: 3月初，用1-3月数据
echo "=== 3月预测（更新） ==="
python train_flexible.py --input_months 3 \
    --save_dir checkpoints/mar

python predict_flexible.py --input_months 3 \
    --stage1_checkpoint checkpoints/mar/stage1_timeseries_best.pth \
    --stage2_checkpoint checkpoints/mar/stage2_yield_best.pth

# 步骤3: 5月初，用1-5月数据（最终预测）
echo "=== 5月预测（高精度） ==="
python train_flexible.py --input_months 5 \
    --save_dir checkpoints/may \
    --epochs_stage1 50

python predict_flexible.py --input_months 5 \
    --stage1_checkpoint checkpoints/may/stage1_timeseries_best.pth \
    --stage2_checkpoint checkpoints/may/stage2_yield_best.pth \
    --visualize
```

---

## 🎯 总结

| 方面 | 说明 |
|------|------|
| **灵活性** | ✅ 支持任意1-11个月的输入 |
| **精度** | ⚠️ 输入越少，精度越低 |
| **推荐配置** | 1-5月（平衡时效和精度） |
| **最小输入** | 建议至少2-3个月 |
| **最佳实践** | 渐进式预测，逐月更新 |

**核心思想**: 
- 早期（2-3月）：快速预警，精度较低
- 中期（4-5月）：平衡点，推荐使用
- 后期（6月+）：高精度，但预测范围小

根据实际需求选择合适的输入长度！

