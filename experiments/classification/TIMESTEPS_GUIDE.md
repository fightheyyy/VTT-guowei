# 不同时间步长度分类训练指南

## 概述

测试不同时间步（数据长度）对早期分类效果的影响。

## 时间步说明

| 时间步 | 天数 | 数据比例 | 说明 |
|--------|------|----------|------|
| 6步    | 60天 | 16% | 极早期识别 |
| 12步   | 120天 | 32% | **推荐** - 平衡信息量和早期性 |
| 18步   | 180天 | 49% | 中期识别 |
| 37步   | 370天 | 100% | 完整序列 |

## 快速开始

### 方法1: 训练12步模型（推荐）

```bash
cd experiments/classification

# Windows - 一键运行
run_12steps.bat

# 或者直接运行（独立版本，不覆盖）
python train_12steps_standalone.py
```

**特点**:
- ✅ 固定使用12步(120天)数据
- ✅ 直接分类，不预测序列
- ✅ 使用Focal Loss处理类别不平衡
- ✅ 双模态(视觉+语言)

**训练配置**:
```python
time_steps = 12
batch_size = 64
epochs = 100
lr = 1e-4
focal_gamma = 2.0  # Focal Loss参数
```

### 方法2: 对比不同时间步

```bash
# 安全版本 - 每个模型独立保存，带时间戳
python compare_timesteps_safe.py
```

会自动训练并对比 6步、12步、18步、37步 四个模型。

**注意**: 这会训练4个模型，需要较长时间（约8-16小时）

**保存位置**: `timesteps_comparison_YYYYMMDD_HHMMSS/` (独立目录，不覆盖)

## 训练逻辑对比

### ✅ 直接分类 (classification文件夹)

```
输入: [batch, 12步, 14变量]
    ↓
TimesCLIPClassifier
    ↓
输出: [batch, 4类别]
```

**特点**:
- 简单直接
- 一个模型，端到端训练
- 逻辑清晰

### ❌ 两阶段预测+分类 (forecasting文件夹)

```
输入: [batch, 6步, 14变量]
    ↓
Forecaster (预测37步)
    ↓
Classifier (分类)
    ↓
输出: [batch, 4类别]
```

**问题**:
- 任务不匹配（目标是分类，不是预测）
- 预测误差累积
- 两个模型，训练复杂

**结论**: 对于分类任务，应该用直接分类！

## 预期结果

### 12步 vs 6步的优势

| 指标 | 6步 | 12步 | 提升 |
|------|-----|------|------|
| F1 Score | ~0.42 | ~0.50-0.55 | +20-30% |
| 信息量 | 少 | 适中 | - |
| 早期性 | 极早 | 早 | - |

**为什么12步更好**:
1. ✅ 更多信息（120天 vs 60天）
2. ✅ 仍然是早期识别（32%数据）
3. ✅ 模式更明显
4. ✅ 减少不确定性

### 12步 vs 37步的对比

| 指标 | 12步 | 37步 | 差异 |
|------|------|------|------|
| F1 Score | ~0.50-0.55 | ~0.60-0.65 | -10-15% |
| 时效性 | 早期(120天) | 晚期(370天) | +250天 |
| 实用价值 | 高 | 低 | - |

**为什么选12步而不是37步**:
- 早期识别才有实用价值
- 10-15%的性能损失换来250天的时间优势
- 符合"早期识别"的任务目标

## 文件说明

### 训练脚本

```
experiments/classification/
├── train_12steps.py           # 12步训练脚本
├── run_12steps.bat            # Windows一键运行
├── compare_timesteps.py       # 对比脚本
└── train_classification_improved.py  # 完整训练脚本
```

### 保存位置（独立保存，互不覆盖）

```
experiments/classification/
├── timesclip_improved/                    # 原有模型
│   └── checkpoints/dual_best.pth
│
├── timesclip_12steps_20241120_143022/     # 12步模型（带时间戳）
│   ├── checkpoints/
│   │   └── best_model.pth                 # 最佳模型
│   ├── results/
│   │   └── test_results.json              # 测试结果
│   └── config.json                        # 训练配置
│
├── timesclip_6steps_20241120_150000/      # 6步模型
│   └── ...
│
└── timesteps_comparison_20241120_160000/  # 对比实验
    ├── timestep_6steps_YYYYMMDD_HHMMSS/
    ├── timestep_12steps_YYYYMMDD_HHMMSS/
    ├── timestep_18steps_YYYYMMDD_HHMMSS/
    ├── timestep_37steps_YYYYMMDD_HHMMSS/
    ├── comparison_summary.json            # 总结果
    └── comparison_plot.png                # 对比图
```

**重要**: 每次训练都会创建带时间戳的新目录，**绝不覆盖**已有模型！

## 使用建议

### 推荐流程

1. **先快速测试12步**
   ```bash
   python train_12steps.py
   ```
   看看效果如何（约2-4小时）

2. **如果效果好，对比其他时间步**
   ```bash
   python compare_timesteps.py
   ```
   找到最佳时间点（约8-16小时）

3. **使用最佳时间步进行完整训练**
   修改`train_12steps.py`中的`time_steps`参数

### 调优建议

如果12步效果不理想：

**方案A: 增加时间步**
```python
time_steps = 18  # 180天
```

**方案B: 增强正则化**
```python
focal_gamma = 3.0  # 更关注难样本
batch_size = 32    # 减小batch增加随机性
```

**方案C: 数据增强**
```python
# 在train_12steps.py中添加
use_data_augmentation = True
```

## 常见问题

### Q1: 12步够吗？信息会不会太少？
**A**: 12步(120天)占总时长的32%，已经足够提取早期模式。农作物生长在前4个月就有明显特征。

### Q2: 为什么不用37步？
**A**: 37步(370天)太晚了，失去了"早期识别"的意义。等到370天再识别，实用价值不大。

### Q3: 6步和12步哪个更好？
**A**: 建议12步。6步太早，信息不足，指标波动大。12步是信息量和早期性的最佳平衡。

### Q4: 训练12步需要多久？
**A**: 
- 每个epoch: ~3-5分钟
- 总训练: ~4-8小时（100 epochs with early stopping）
- 取决于GPU性能

### Q5: 12步的F1能达到多少？
**A**: 
- 预期: 0.50-0.55
- 比6步(~0.42)提升20-30%
- 比37步(~0.60-0.65)低10-15%

### Q6: 可以同时训练多个时间步吗？
**A**: 不建议。串行训练，保证每个模型都充分训练。用`compare_timesteps.py`自动完成。

## 总结

### 为什么选择12步直接分类？

1. ✅ **任务匹配**: 目标是分类，不是预测序列
2. ✅ **信息充足**: 120天数据，模式明显
3. ✅ **仍然早期**: 只用32%数据
4. ✅ **逻辑简单**: 一个模型，端到端
5. ✅ **训练稳定**: 不像forecasting那样波动

### 核心优势

- 相比forecasting的两阶段：**逻辑更合理**
- 相比6步：**信息更充分，指标更稳定**
- 相比37步：**早期性更好，实用价值更高**

**开始训练**: 直接运行 `run_12steps.bat` 即可！

