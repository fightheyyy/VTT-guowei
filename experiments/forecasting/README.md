# TimesCLIP序列预测 - 两阶段早期识别方案

## 方案概述

将早期识别问题分解为两个子任务：
1. **序列预测**：从早期部分序列（如前6步）预测完整序列（37步）
2. **序列分类**：对预测的完整序列进行分类

## 目录结构

```
experiments/forecasting/
├── train_forecaster.py        # 单独训练预测器
├── two_stage_pipeline.py      # 两阶段流程训练
├── checkpoints/               # 模型保存目录
└── results/                   # 结果可视化目录
```

## 快速开始

### 1. 测试模型架构（无需训练）

```bash
# 测试预测器
cd VTT-guowei
python models/timesclip_forecaster.py
```

### 2. 训练序列预测器（阶段1）

```bash
# MLP解码器（推荐，快速）
python experiments/forecasting/train_forecaster.py \
    --input_len 6 \
    --decoder_type mlp \
    --epochs 50

# LSTM解码器（考虑时间依赖）
python experiments/forecasting/train_forecaster.py \
    --input_len 6 \
    --decoder_type lstm \
    --epochs 50

# Transformer解码器（最强大，但最慢）
python experiments/forecasting/train_forecaster.py \
    --input_len 6 \
    --decoder_type transformer \
    --epochs 50
```

### 3. 两阶段独立训练

```bash
# 先训练预测器，再训练分类器（预测器冻结）
python experiments/forecasting/two_stage_pipeline.py \
    --mode stage2 \
    --input_len 6 \
    --epochs 50
```

### 4. 端到端联合训练（推荐）

```bash
# 预测和分类联合优化
python experiments/forecasting/two_stage_pipeline.py \
    --mode e2e \
    --input_len 6 \
    --epochs 100
```

## 训练策略对比

| 策略 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| **单独训练预测器** | 简单，可独立评估 | 不考虑分类任务 | 快速原型验证 |
| **两阶段独立** | 模块化，稳定 | 预测误差传播 | 预测器已预训练 |
| **端到端联合** | 性能最优，误差可优化 | 训练复杂 | 追求最佳性能 |

## 超参数说明

### 预测器参数

```python
input_len=6          # 输入序列长度（6步=60天）
output_len=37        # 输出序列长度（37步=370天）
decoder_type='mlp'   # 解码器类型：mlp/lstm/transformer
use_vision=False     # 是否使用视觉分支（预测任务通常不需要）
use_language=True    # 是否使用语言分支
patch_length=2       # patch大小（短序列用小patch）
stride=1             # patch步长（滑动窗口）
```

### 训练参数

```python
batch_size=64        # 批次大小
epochs=100           # 训练轮数
lr=1e-4              # 学习率
alpha=0.3            # 端到端训练时预测损失权重
```

## 解码器选择建议

### MLP解码器
- **优点**：最快，参数少，适合快速实验
- **缺点**：不考虑时间依赖，预测质量可能较差
- **适用**：baseline，快速原型

### LSTM解码器
- **优点**：考虑时间依赖，自回归预测
- **缺点**：较慢，需要逐步预测
- **适用**：需要捕捉时间动态

### Transformer解码器
- **优点**：最强大，并行预测，长期依赖
- **缺点**：最慢，参数最多，可能过拟合
- **适用**：数据充足，追求最优

## 预期效果

基于初步估计：

| 输入长度 | 直接分类F1 | 两阶段F1（独立） | 两阶段F1（端到端） |
|---------|-----------|-----------------|------------------|
| 6步(60天) | 0.10 | **0.35** | **0.45** |
| 9步(90天) | 0.10 | **0.45** | **0.55** |
| 12步(120天) | 0.15 | **0.55** | **0.65** |

## 关键改进点

### 1. Patch策略优化
```python
# 原模型（不适合短序列）
patch_length=4, stride=4  
→ 6步只有1个patch

# 改进模型
patch_length=2, stride=1  
→ 6步有5个patch  ✓
```

### 2. 双模态选择
```python
# 预测任务：语言分支足够
use_vision=False   # 减少计算，避免信息损失
use_language=True
```

### 3. 联合训练损失
```python
loss = alpha * MSE(预测) + (1-alpha) * CE(分类)
# alpha动态调整：早期重预测，后期重分类
```

## 可视化结果

训练后会生成：
- `results/predictions_{decoder}_in{input_len}.png` - 预测序列可视化
- `results/loss_curve_{decoder}_in{input_len}.png` - 训练曲线

## 测试不同时间尺度

```bash
# 测试6步、9步、12步、15步的效果
for len in 6 9 12 15; do
    python experiments/forecasting/two_stage_pipeline.py \
        --mode e2e \
        --input_len $len \
        --epochs 100
done
```

## 故障排除

### 1. CUDA内存不足
- 减小 `batch_size`
- 使用 `decoder_type='mlp'`
- 设置 `use_vision=False`

### 2. 预测质量差
- 增加训练轮数
- 尝试不同解码器
- 调整 `patch_length` 和 `stride`

### 3. 分类效果没提升
- 检查预测MSE是否合理
- 调整端到端训练的 `alpha`
- 尝试先预训练预测器

## 下一步优化方向

1. **数据增强**：时间扭曲、噪声注入
2. **多任务学习**：同时预测多个时间尺度
3. **不确定性建模**：贝叶斯预测 + 概率分类
4. **模型轻量化**：知识蒸馏、剪枝

