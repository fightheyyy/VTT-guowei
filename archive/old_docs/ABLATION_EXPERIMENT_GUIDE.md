# 消融实验指南：对比双模态 vs 语言模态

## 实验目的

验证**视觉模态**和**对比学习对齐**在模型中的实际作用，回答以下问题：
1. 视觉模态能提升多少性能？
2. 对比学习的对齐损失是否必要？
3. 去掉视觉模态能加速多少？

---

## 实验设置

### 对照组（原始双模态）
- **文件**: `train_multiyear_mirror.py`
- **模型**: `TimesCLIP` (视觉 + 语言 + 对比对齐)
- **特点**:
  - 使用 matplotlib 生成时序图像
  - CLIP 编码图像特征
  - InfoNCE 对比损失对齐视觉和语言模态
  - 训练较慢（图像生成开销大）

### 实验组（语言模态版本）
- **文件**: `train_language_only.py`
- **模型**: `TimesCLIPLanguageOnly` (仅语言模态)
- **特点**:
  - 无视觉预处理和视觉模块
  - 无对比学习对齐
  - 只用 BERT 编码 patch 序列
  - 训练快速（无图像生成）

### 控制变量
为了公平对比，两个版本保持以下参数一致：
- 数据集：相同的训练/测试集
- Batch Size：8
- Epochs：阶段1=100，阶段2=30
- 学习率：相同
- d_model：256
- patch_length：6，stride：3

---

## 使用方法

### 步骤1：运行双模态版本（对照组）

在 1080Ti 电脑上运行：
```bash
python train_multiyear_mirror.py
```

**输出**:
- `checkpoints/stage1_timeseries_best.pth`
- `checkpoints/stage2_yield_best.pth`
- `logs/stage1/`
- `logs/stage2/`

**预计时间**: 
- 阶段1: 约 8-10 小时（100 epochs × 5分钟/epoch）
- 阶段2: 约 2-3 小时（30 epochs）

---

### 步骤2：运行语言模态版本（实验组）

在同一台电脑上运行：
```bash
python train_language_only.py
```

**输出**:
- `checkpoints/stage1_language_only_best.pth`
- `checkpoints/stage2_language_only_best.pth`
- `logs/stage1_language_only/`
- `logs/stage2_language_only/`

**预计时间**: 
- 阶段1: 约 2-3 小时（100 epochs × 1-2分钟/epoch）⚡ **快3-5倍**
- 阶段2: 约 2-3 小时（相同）

**为什么快这么多？**
- 不需要生成 56 张图像（8 batch × 7 变量）
- 不需要 CLIP 视觉编码
- 不需要计算对齐损失的大矩阵

---

### 步骤3：对比结果

```bash
python compare_results.py
```

**输出示例**:
```
======================================================================
对比总结
======================================================================

阶段1（时间序列补全）:
  指标              双模态          语言模态        差异
  ────────────────────────────────────────────────────────
  MSE Loss:        0.178000     0.185000     +3.9%
  RMSE:            0.4220       0.4301       +1.9%
  MAE:             0.3180       0.3250       +2.2%
  参数量:          25,648,128   18,432,000   -28.1%

  性能保留率: 98.1% (RMSE)
  结论: ⭐ 语言模态性能相当（差异<5%），建议去掉视觉模态

阶段2（产量预测）:
  指标              双模态          语言模态        差异
  ────────────────────────────────────────────────────────
  MSE Loss:        0.523000     0.531000     +1.5%
  RMSE:            0.7232       0.7287
  R² Score:        0.8245       0.8198       -0.6%
```

**结果保存在**: `comparison_results.json`

---

## 预期结果与判断标准

### 情况1: 性能相当（差异 < 5%）⭐
```
RMSE 差异: +2%
R² 差异: -0.5%
```
**结论**: 视觉模态的贡献很小，建议：
- ✅ 采用语言模态版本
- ✅ 训练速度提升 3-5 倍
- ✅ 参数量减少 30%
- ✅ 性能几乎不损失

### 情况2: 略有差异（差异 5-10%）⚠
```
RMSE 差异: +7%
R² 差异: -4%
```
**结论**: 视觉模态有一定作用，需要权衡：
- 如果看重速度 → 用语言模态版本
- 如果看重性能 → 保留双模态，但优化视觉预处理（如缓存图像）

### 情况3: 显著差异（差异 > 10%）❌
```
RMSE 差异: +15%
R² 差异: -8%
```
**结论**: 视觉模态很重要，应该保留：
- ❌ 不建议去掉视觉模态
- ✅ 但可以优化加速（见下文）

---

## 训练过程对比

### 双模态版本
```
Epoch 1/100: 100%|████| 187/187 [05:16<00:00, 1.69s/it, loss=0.785, gen=0.394, align=3.9]
Epoch 1/100:
  Train Loss: 0.939586 (Gen: 0.538376, Align: 4.012097)
  Val Loss:   0.753231
```
- 每个 epoch 约 5 分钟
- 有两个损失：gen_loss + align_loss
- align_loss 从 4.0 降到 1.2

### 语言模态版本
```
Epoch 1/100: 100%|████| 187/187 [01:05<00:00, 0.35s/it, loss=0.542]
Epoch 1/100:
  Train Loss: 0.542138 (RMSE: 0.7363)
  Val Loss:   0.518245 (RMSE: 0.7199)
```
- 每个 epoch 约 1 分钟 ⚡ **快 5 倍**
- 只有一个损失：mse_loss
- 直接显示 RMSE，更直观

---

## TensorBoard 可视化

### 查看训练曲线
```bash
tensorboard --logdir=logs
```

访问 `http://localhost:6006`，对比：

**双模态**:
- `logs/stage1/` - Loss/train, Loss/val, Loss/gen, Loss/align
- `logs/stage2/` - Loss/train, Loss/val, Metrics/R2

**语言模态**:
- `logs/stage1_language_only/` - Loss/train, Loss/val, RMSE/train, RMSE/val
- `logs/stage2_language_only/` - Loss/train, Loss/val, Metrics/R2

---

## 进一步分析

### 查看参数量
```python
# 双模态版本
python -c "
from models import TimesCLIP
model = TimesCLIP(time_steps=18, n_variates=7, prediction_steps=18, d_model=256)
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
"

# 语言模态版本
python -c "
from models.timesclip_language_only import TimesCLIPLanguageOnly
model = TimesCLIPLanguageOnly(time_steps=18, n_variates=7, prediction_steps=18, d_model=256)
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
"
```

### 测试推理速度
```python
import torch
import time
from models import TimesCLIP
from models.timesclip_language_only import TimesCLIPLanguageOnly

# 准备数据
x = torch.randn(8, 18, 7).cuda()

# 测试双模态
model_both = TimesCLIP(time_steps=18, n_variates=7, prediction_steps=18, d_model=256).cuda()
model_both.eval()

start = time.time()
with torch.no_grad():
    for _ in range(100):
        y_pred, _ = model_both(x)
time_both = time.time() - start

# 测试语言模态
model_language = TimesCLIPLanguageOnly(time_steps=18, n_variates=7, prediction_steps=18, d_model=256).cuda()
model_language.eval()

start = time.time()
with torch.no_grad():
    for _ in range(100):
        y_pred = model_language(x)
time_language = time.time() - start

print(f"双模态: {time_both:.2f}s")
print(f"语言模态: {time_language:.2f}s")
print(f"加速比: {time_both / time_language:.1f}x")
```

---

## 常见问题

### Q1: 为什么语言模态版本也用了 BERT？
**A**: BERT 是语言模态的核心，我们去掉的是视觉模态（CLIP + 图像生成）。BERT 对 patch 序列编码是必要的。

### Q2: 如果性能相当，为什么不直接用简单的 Transformer？
**A**: 可以进一步尝试。当前的语言模态版本仍然用了预训练的 BERT，下一步可以测试完全从头训练的 Transformer。

### Q3: 训练到一半想停止怎么办？
**A**: 
- 按 `Ctrl+C` 停止
- 已保存的 checkpoint 在 `checkpoints/stage1_*_epoch_*.pth`
- 可以修改代码从 checkpoint 继续训练

### Q4: 1080Ti 显存不够怎么办？
**A**: 两个版本都已经用了 `batch_size=8`，如果还不够：
```python
# 减小 batch size
batch_size=4

# 或者减小模型
d_model=128  # 从 256 减到 128
```

---

## 实验记录模板

将结果填入下表：

| 指标 | 双模态 | 语言模态 | 差异 | 判断 |
|------|--------|----------|------|------|
| **阶段1** | | | | |
| MSE Loss | | | | |
| RMSE | | | | |
| 参数量 | | | | |
| 训练时间 | | | | |
| **阶段2** | | | | |
| MSE Loss | | | | |
| R² Score | | | | |
| 训练时间 | | | | |
| **总结** | | | | |
| 性能保留率 | - | | | |
| 速度提升 | - | | | |
| **建议** | | | | |

---

## 下一步

根据实验结果：

### 如果性能相当（推荐）
1. ✅ 采用 `train_language_only.py` 作为主训练脚本
2. ✅ 进一步测试不用 BERT 的纯 Transformer 版本
3. ✅ 增大 batch_size 到 16 或 32（因为省显存了）
4. ✅ 尝试更深的网络或更多的 epochs

### 如果视觉模态重要
1. ⚠ 保留双模态架构
2. ✅ 优化视觉预处理（缓存图像、用更快的绘图库）
3. ✅ 优化对齐损失（改进正负样本比例）
4. ✅ 尝试其他视觉编码器（更轻量的 CNN）

---

## 总结

这个消融实验能够：
- ✅ 量化视觉模态的实际贡献
- ✅ 发现性能瓶颈和优化方向
- ✅ 在性能和效率间做出明智的权衡

**预测**：语言模态版本的性能应该在 95-98% 左右，如果属实，强烈建议采用。

祝实验顺利！

