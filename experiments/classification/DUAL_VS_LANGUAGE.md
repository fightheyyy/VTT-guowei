# 双模态 vs 纯语言 - 12步直接分类

## 🎯 两种训练方案

### 方案1: 纯语言模态
**脚本**: `train_12steps_language_only.py`

```
输入: [batch, 12步, 14变量]
    ↓
┌─────────────────────────┐
│ CLIP-Text Encoder       │
│ (冻结)                   │
└─────────────────────────┘
    ↓
语言特征 [batch, 512]
    ↓
┌─────────────────────────┐
│ 分类头                   │
│ 512 → 256 → 4           │
└─────────────────────────┘
    ↓
输出: [batch, 4类别]
```

**特点**:
- ✅ 训练快（无需处理图像）
- ✅ 显存占用小
- ✅ 逻辑简单
- ⚠️ 只用语言模态

**适合**: 快速验证12步效果

---

### 方案2: 真双模态 + 对比学习 ⭐ [创新点]
**脚本**: `train_12steps_dual_cached.py`

```
输入: [batch, 12步, 14变量] + 预缓存图像
    ↓
┌──────────────────┐    ┌──────────────────┐
│ CLIP-Vision      │    │ CLIP-Text        │
│ (冻结)           │    │ (冻结)           │
└──────────────────┘    └──────────────────┘
     ↓                       ↓
  视觉特征                语言特征
  [batch, 512]           [batch, 512]
     ↓                       ↓
     └────── InfoNCE ────────┘
     ↓                       ↓
┌─────────────────────────────────┐
│ 变量选择 + 特征融合              │
└─────────────────────────────────┘
     ↓
融合特征 [batch, 1024]
     ↓
┌─────────────────────────┐
│ 分类头                   │
│ 1024 → 256 → 4          │
└─────────────────────────┘
     ↓
输出: [batch, 4类别]
```

**特点**:
- ✅ 视觉 + 语言双模态
- ✅ InfoNCE对比学习
- ✅ 特征对齐和融合
- ✅ **这是你的创新点！**
- ⚠️ 需要预缓存图像
- ⚠️ 训练稍慢

**适合**: 发论文，展示创新

---

## 📊 详细对比

| 项目 | 纯语言 | **真双模态** |
|------|--------|-------------|
| 视觉分支 | ❌ | ✅ |
| 语言分支 | ✅ | ✅ |
| 对比学习 | ❌ | ✅ InfoNCE |
| 图像数据 | 无 | ✅ 预缓存 |
| 训练速度 | 快 | 中等 |
| 显存占用 | 小 (~6GB) | 中 (~10GB) |
| 预期F1 | ~0.50 | **~0.52-0.55** |
| 参数量 | ~65M | ~131M |
| 可训练参数 | ~4M | ~8M |
| 创新点 | ❌ | **✅** |

---

## 🚀 如何使用

### 方法1: 使用批处理文件

```bash
cd experiments/classification
.\run_12steps.bat

# 选择:
# [1] 纯语言模态
# [2] 真双模态 + 对比学习
```

### 方法2: 直接运行Python

```bash
# 纯语言
python train_12steps_language_only.py

# 真双模态
python train_12steps_dual_cached.py
```

---

## 📁 保存位置（互不覆盖）

### 纯语言模型
```
experiments/classification/timesclip_12steps_language_YYYYMMDD_HHMMSS/
├── checkpoints/
│   └── best_model.pth
├── results/
│   └── test_results.json
└── config.json
    {
      "model_type": "language_only",
      "use_contrastive": false,
      ...
    }
```

### 双模态模型
```
experiments/classification/timesclip_12steps_dual_YYYYMMDD_HHMMSS/
├── checkpoints/
│   └── best_model.pth
├── results/
│   └── test_results.json
└── config.json
    {
      "model_type": "dual",
      "use_contrastive": true,
      "image_cache": "time_12",
      ...
    }
```

---

## ⚙️ 图像缓存说明

### 双模态需要的图像缓存

```
data/multiscale_image_cache/time_12/
├── sample_0_variate_0.png
├── sample_0_variate_1.png
├── ...
├── sample_0_variate_13.png
├── sample_1_variate_0.png
├── ...
└── sample_5556_variate_13.png

总数: 5557样本 × 14变量 = 77,798张图
```

### 检查图像缓存

```python
import os
cache_dir = "data/multiscale_image_cache/time_12"
files = os.listdir(cache_dir)
print(f"图像数量: {len(files)}")
print(f"预期: {5557 * 14}")
```

### 如果缺少图像

运行图像生成脚本:
```bash
cd experiments/classification
python prepare_multiscale_images.py --time_steps 12
```

---

## 💡 建议的训练流程

### 阶段1: 快速验证（1-2天）

```bash
# 先用纯语言验证12步是否比6步好
python train_12steps_language_only.py
```

**目的**: 快速得到结果，验证12步的有效性

### 阶段2: 完整训练（2-3天）

```bash
# 如果纯语言效果好，再训练双模态
python train_12steps_dual_cached.py
```

**目的**: 发挥双模态优势，获得最佳效果

### 阶段3: 对比分析

对比两个模型的结果，说明双模态的提升。

---

## 📈 预期结果

### 纯语言模态

```
训练时间: 2-4小时
F1 Score: 0.48-0.52
准确率: 0.53-0.57

各类别F1:
  Class 0: 0.32-0.38
  Class 1: 0.45-0.50
  Class 2: 0.50-0.55
  Class 3: 0.60-0.65
```

### 真双模态

```
训练时间: 4-6小时
F1 Score: 0.52-0.56 (+4-8% vs 纯语言)
准确率: 0.57-0.61

各类别F1:
  Class 0: 0.38-0.42 (少数类提升明显)
  Class 1: 0.48-0.53
  Class 2: 0.54-0.58
  Class 3: 0.62-0.67

双模态优势:
  - 对比学习对齐视觉和语言特征
  - 视觉提供额外判别信息
  - 少数类识别提升更明显
```

---

## 🎓 论文写作建议

### 消融实验

训练两个模型，展示双模态的贡献：

```
Table: Ablation Study on 12-step Early Recognition

| Model | Visual | Language | Contrastive | F1 (macro) |
|-------|--------|----------|-------------|------------|
| Language-only | ❌ | ✅ | ❌ | 0.50 |
| **Dual-modal** | ✅ | ✅ | ✅ | **0.54** |

Improvement: +8% F1 with dual-modal design
```

### 创新点阐述

1. **早期识别** - 仅用12步(120天，32%数据)
2. **双模态融合** - 视觉+语言互补
3. **对比学习** - InfoNCE对齐特征
4. **直接分类** - 无需预测完整序列

### 可视化

- 对比学习的特征对齐可视化
- 不同时间步的性能曲线
- 混淆矩阵对比
- 注意力权重可视化

---

## ❓ 常见问题

### Q1: 为什么双模态更好？
**A**: 
- 视觉提供全局模式（趋势、周期）
- 语言提供局部细节（数值、变化）
- 对比学习让两者互补

### Q2: 纯语言够用吗？
**A**: 
- 如果只是快速实验，够用
- 如果要发论文，建议用双模态展示创新

### Q3: 图像缓存怎么生成？
**A**: 
```bash
python prepare_multiscale_images.py --time_steps 12
```

### Q4: 可以不用缓存，动态生成图像吗？
**A**: 
- 可以，但非常慢（每个batch都要画图）
- 不推荐，影响训练效率

### Q5: 双模态比纯语言提升多少？
**A**: 
- 预期F1提升4-8%
- 少数类提升更明显（10-15%）

---

## 🎯 总结

### 推荐流程

```
1. 先训练纯语言 → 快速验证12步有效性
2. 再训练双模态 → 获得最佳效果
3. 对比两者 → 展示双模态优势
4. 写论文 → 强调创新点
```

### 核心优势

- ✅ 12步早期识别
- ✅ 双模态创新
- ✅ 对比学习
- ✅ 独立保存，可复现

**现在可以开始训练了！运行 `.\run_12steps.bat` 选择模式。**

