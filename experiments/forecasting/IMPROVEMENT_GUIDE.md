# 处理类别不平衡数据集的改进指南

## 数据集问题总结

### 2018four.csv 存在的问题

1. **严重类别不平衡** (不平衡比例: 3.56x)
   ```
   Class 0:  671 样本 (12.07%)  ← 最少
   Class 1: 1210 样本 (21.77%)
   Class 2: 1288 样本 (23.18%)
   Class 3: 2388 样本 (42.97%)  ← 最多
   ```

2. **验证集较小**
   - 总验证样本: 834
   - Class 0只有100个验证样本
   - 导致指标波动大

3. **任务难度高**
   - 从6步预测31步
   - 信息不足，容易过拟合

---

## 改进策略

### 核心改进 (必须使用)

#### 1. Focal Loss
**作用**: 让模型更关注难分类样本和少数类

**原理**:
```python
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- `α`: 类别权重 (少数类权重大)
- `γ`: 聚焦参数 (通常2.0)
- `p_t`: 预测概率

**效果**: 
- 减少多数类(Class 3)对损失的主导
- 提升少数类(Class 0)的学习效果

**代码**:
```python
class_weights = torch.FloatTensor([2.069, 1.148, 1.079, 0.582])
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

#### 2. 类别权重
**作用**: 平衡各类别对损失的贡献

**计算方式**:
```python
weight_i = total_samples / (num_classes × samples_i)
```

**当前数据集的权重**:
```
Class 0: 2.069 (样本最少，权重最大)
Class 1: 1.148
Class 2: 1.079
Class 3: 0.582 (样本最多，权重最小)
```

#### 3. 增强正则化
**配置**:
```python
dropout = 0.3        # 从0.1提升到0.3
weight_decay = 1e-3  # 从1e-4提升到1e-3
```

**作用**: 
- 防止模型记住训练样本
- 提升泛化能力

#### 4. 降低学习率
```python
lr = 5e-5  # 从1e-4降到5e-5
```

**作用**: 更细致的优化，避免震荡

---

### 辅助改进 (推荐使用)

#### 5. 数据增强
**方法**:
- 添加噪音
- 随机缩放
- 时间平移
- 变量dropout

**代码**:
```python
# 在训练时自动应用
train_dataset = ForecastingDataset(
    X_train, input_len, output_len, 
    augment=True  # 开启增强
)
```

#### 6. 平衡采样
**作用**: 让每个batch中各类样本数量相近

```python
train_sampler = ImbalancedDatasetSampler(train_dataset, y_train)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    sampler=train_sampler  # 使用平衡采样器
)
```

**注意**: 平衡采样和类别权重二选一即可

#### 7. 梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**作用**: 防止梯度爆炸，稳定训练

#### 8. 更小的Batch Size
```python
batch_size = 32  # 从64降到32
```

**作用**: 增加随机性，提升泛化

---

## 使用方法

### 方法1: 快速启动 (推荐)

```bash
cd experiments/forecasting
run_improved.bat
```

选择 `[1] 端到端Pipeline` 即可开始训练。

### 方法2: 命令行

```bash
cd experiments/forecasting

# 训练改进版Pipeline
python train_pipeline_improved.py \
    --csv_path ../../data/2018four.csv \
    --input_len 6 \
    --batch_size 32 \
    --epochs 100 \
    --lr 5e-5 \
    --dropout 0.3 \
    --focal_gamma 2.0
```

### 方法3: 继续训练 (从checkpoint恢复)

```bash
python train_pipeline_improved.py \
    --checkpoint checkpoints/pipeline_e2e_in6.pth \
    --epochs 100 \
    --lr 5e-5
```

这会：
- 加载已训练的模型
- 继续训练到100 epochs
- 保留之前的最佳F1记录

---

## 预期效果

### 原版性能 (当前)
```
Best Val F1: ~0.42
波动范围: 0.27 - 0.42
问题: 不稳定，少数类预测差
```

### 改进版预期性能
```
Best Val F1: 0.50 - 0.55 (提升20-30%)
波动范围: 0.48 - 0.55
改善: 更稳定，少数类F1提升
```

### 各类别F1预期改善
| 类别 | 原版F1 | 改进版F1 | 提升 |
|------|--------|----------|------|
| Class 0 | ~0.15 | ~0.35 | +133% |
| Class 1 | ~0.40 | ~0.50 | +25% |
| Class 2 | ~0.45 | ~0.55 | +22% |
| Class 3 | ~0.70 | ~0.65 | -7% |

**注意**: Class 3的F1可能略降，因为不再过度拟合多数类。但整体macro F1提升。

---

## 关键参数调优指南

### 如果验证F1仍然波动大

**方案A: 增强正则化**
```python
dropout = 0.4  # 进一步提高
weight_decay = 5e-3
```

**方案B: 降低学习率**
```python
lr = 2e-5  # 更慢但更稳定
```

**方案C: 增加Focal Loss的gamma**
```python
focal_gamma = 3.0  # 更关注难样本
```

### 如果训练损失不下降

**方案A: 提高学习率**
```python
lr = 1e-4  # 回到原值
```

**方案B: 降低正则化**
```python
dropout = 0.2
weight_decay = 5e-4
```

### 如果Class 0的F1仍然很低

**方案A: 增加Class 0的权重**
```python
# 手动调整类别权重
class_weights = torch.FloatTensor([3.0, 1.148, 1.079, 0.582])
```

**方案B: 使用过采样**
```python
--use_balanced_sampling  # 启用平衡采样
```

**方案C: 数据增强**
```python
--use_augmentation  # 启用数据增强
```

---

## 训练监控

### 良好的训练曲线应该是:
```
Epoch 1:  Loss=1.2, Val F1=0.35
Epoch 10: Loss=0.9, Val F1=0.42
Epoch 20: Loss=0.8, Val F1=0.48
Epoch 30: Loss=0.7, Val F1=0.51
Epoch 40: Loss=0.7, Val F1=0.52 ← 最佳
Epoch 50: Loss=0.6, Val F1=0.51
```

**特征**:
- 训练损失稳定下降
- 验证F1稳定上升，然后平稳
- 波动范围小 (±0.03以内)

### 如果出现以下情况需要调整:

**情况1: 验证F1剧烈波动** (0.3 → 0.5 → 0.3)
→ 增加dropout, 降低学习率

**情况2: 训练损失很低但验证F1低** (Loss=0.3, F1=0.35)
→ 严重过拟合，增加正则化

**情况3: 训练和验证损失都不下降**
→ 学习率太低或模型容量不足

---

## 对比实验

建议运行对比实验，验证改进效果:

```bash
# 自动运行对比
run_improved.bat -> 选择 [4]
```

这会：
1. 训练原版模型 (50 epochs)
2. 训练改进版模型 (50 epochs)
3. 对比两者的F1和稳定性

---

## 常见问题

### Q1: 为什么还是会过拟合?
**A**: 数据集本身样本有限。改进策略能缓解但不能完全消除。可以：
- 收集更多数据
- 使用更简单的模型
- 更强的正则化

### Q2: 类别权重和Focal Loss可以同时用吗?
**A**: 可以! Focal Loss中的α参数就是类别权重。当前实现已经结合了两者。

### Q3: 数据增强会让训练变慢吗?
**A**: 会略微变慢(~5-10%)，但效果提升明显，非常值得。

### Q4: 为什么不用SMOTE等过采样方法?
**A**: 
- 时间序列数据的SMOTE较复杂
- 平衡采样已经能起到类似效果
- 数据增强更适合时间序列

### Q5: 能达到F1=0.7吗?
**A**: 在当前数据集上较难。主要限制因素:
- 类别不平衡严重
- 样本量有限
- 任务难度高 (6步预测31步)

现实预期: F1=0.50-0.55已经是很好的结果。

---

## 总结

### 最重要的3个改进:
1. ✅ **Focal Loss + 类别权重** - 处理不平衡的核心
2. ✅ **增强正则化** - 防止过拟合
3. ✅ **降低学习率** - 稳定训练

### 建议的完整配置:
```python
batch_size = 32
lr = 5e-5
dropout = 0.3
weight_decay = 1e-3
focal_gamma = 2.0
use_focal_loss = True
use_data_augmentation = True
clip_grad_norm = 1.0
```

### 预期训练时间:
- 每个epoch: ~2小时 (取决于GPU)
- 总训练时间: ~50-100小时
- 建议: 使用checkpoint恢复功能，分批训练

---

**开始训练**: 直接运行 `run_improved.bat` 选择选项1即可!

