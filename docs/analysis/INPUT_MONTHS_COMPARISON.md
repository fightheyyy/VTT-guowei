# 输入月份对比速查表

## 🎯 快速选择指南

**问题**: 我有多少个月的数据？该如何配置？

| 你的情况 | 输入月份 | 命令 | 精度 | 何时使用 |
|---------|---------|------|------|---------|
| 2月底，只有1-2月数据 | 2 | `--input_months 2` | ⭐⭐ | 需要最早预警 |
| 3月底，有1-3月数据 | 3 | `--input_months 3` | ⭐⭐⭐ | 早期评估 |
| 4月底，有1-4月数据 | 4 | `--input_months 4` | ⭐⭐⭐⭐ | 较准确评估 |
| 5月底，有1-5月数据 | 5 | `--input_months 5` | ⭐⭐⭐⭐⭐ | **推荐配置** |
| 6月底，有1-6月数据 | 6 | `--input_months 6` | ⭐⭐⭐⭐⭐+ | 高精度预测 |

---

## 📊 详细对比

### 配置对比表

| 配置项 | 2个月 | 3个月 | 4个月 | 5个月 (推荐) | 6个月 |
|-------|-------|-------|-------|-------------|-------|
| **输入时间步** | 6 | 9 | 12 | 15-18 | 18 |
| **预测时间步** | 30 | 27 | 24 | 18-21 | 18 |
| **预测难度** | 很高 | 高 | 中高 | 中等 | 较低 |
| **训练时间** | 快 | 快 | 中等 | 中等 | 稍慢 |
| **推荐epochs** | 30 | 40 | 50 | 50 | 60 |
| **推荐d_model** | 128 | 256 | 256 | 256 | 512 |

### 精度影响

```
输入1-2月  ████░░░░░░  20% 信息 → 精度较低
输入1-3月  ██████░░░░  40% 信息 → 精度中等
输入1-4月  ████████░░  60% 信息 → 精度较好
输入1-5月  ██████████  80% 信息 → 精度很好 ⭐
输入1-6月  ██████████  100% 信息 → 精度优秀
```

---

## 🚀 使用示例

### 场景1: 最早预测（2月）

```bash
# 训练
python train_flexible.py \
    --input_months 2 \
    --epochs_stage1 30 \
    --d_model 128 \
    --batch_size 32

# 预测
python predict_flexible.py \
    --input_months 2 \
    --visualize
```

**预期**: 能给出大致趋势，但数值可能偏差较大

### 场景2: 平衡预测（5月）⭐

```bash
# 训练
python train_flexible.py \
    --input_months 5 \
    --epochs_stage1 50 \
    --d_model 256

# 预测
python predict_flexible.py \
    --input_months 5 \
    --visualize
```

**预期**: 较好的精度，推荐使用

### 场景3: 高精度预测（6月+）

```bash
# 训练
python train_flexible.py \
    --input_months 6 \
    --epochs_stage1 60 \
    --d_model 512

# 预测
python predict_flexible.py \
    --input_months 6 \
    --visualize
```

**预期**: 高精度，但预测范围缩小

---

## 💡 选择建议

### 按需求选择

**如果需要早期预警** (2-3月)
- 接受较低精度
- 作为参考值
- 后续逐月更新

**如果需要准确预测** (5-6月)
- 较高精度
- 可靠性强
- 适合决策依据

### 按数据可用性选择

```python
def choose_config(available_months):
    if available_months <= 2:
        return "最早预测，精度有限"
    elif available_months <= 4:
        return "早期预测，可作参考"
    elif available_months <= 5:
        return "推荐配置，平衡点"
    else:
        return "高精度预测"
```

---

## 📈 渐进式预测策略

**最佳实践**: 随数据累积逐月更新预测

```
2月底 → 用2个月数据 → 初步预测（参考）
   ↓
3月底 → 用3个月数据 → 更新预测（较准）
   ↓
4月底 → 用4个月数据 → 再次更新（更准）
   ↓
5月底 → 用5个月数据 → 最终预测（推荐）
```

### 实现代码

```bash
#!/bin/bash
# 渐进式预测脚本

for months in 2 3 4 5 6; do
    echo "=== 训练 ${months}个月配置 ==="
    python train_flexible.py \
        --input_months $months \
        --save_dir checkpoints/months_${months}
    
    echo "=== 预测 ==="
    python predict_flexible.py \
        --input_months $months \
        --stage1_checkpoint checkpoints/months_${months}/stage1_timeseries_best.pth \
        --stage2_checkpoint checkpoints/months_${months}/stage2_yield_best.pth
done
```

---

## ⚠️ 重要提醒

### 1. 训练和预测匹配

```bash
# ✅ 正确
python train_flexible.py --input_months 3
python predict_flexible.py --input_months 3

# ❌ 错误
python train_flexible.py --input_months 3
python predict_flexible.py --input_months 5  # 不匹配！
```

### 2. 时间步验证

```python
# 必须满足
lookback + prediction_steps = 36

# 示例
input_months = 3
lookback = 3 * 3 = 9
prediction_steps = 36 - 9 = 27  ✓
```

### 3. 最小输入建议

- **理论最小**: 1个月（3步）
- **实践最小**: 2个月（6步）
- **推荐最小**: 3个月（9步）

---

## 🔬 实验对比

建议做消融实验，对比不同输入长度的效果：

| 指标 | 2个月 | 3个月 | 4个月 | 5个月 | 6个月 |
|------|-------|-------|-------|-------|-------|
| 阶段1 MSE | ? | ? | ? | ? | ? |
| 阶段2 MAE | ? | ? | ? | ? | ? |
| 总训练时间 | ? | ? | ? | ? | ? |
| 推理速度 | ? | ? | ? | ? | ? |

填写实验结果后，可以为具体数据集选择最优配置。

---

## 📝 总结

| 方面 | 说明 |
|------|------|
| **灵活性** | ✅ 支持2-11个月任意输入 |
| **权衡** | 时效性 ⇄ 精度 |
| **推荐** | 5个月（最佳平衡点） |
| **最小** | 2-3个月（可用但精度低） |
| **最优** | 6个月及以上（高精度） |
| **策略** | 渐进式预测，逐月更新 |

**核心原则**: 根据实际数据可用性和精度要求灵活选择！

