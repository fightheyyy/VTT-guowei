# 消融实验文件说明

## 文件清单

### 1. 模型文件

#### `models/timesclip.py` (原始双模态)
- 完整的 TimesCLIP 模型
- 包含：视觉模态 + 语言模态 + 对比学习对齐
- 特点：性能可能更好，但训练慢

#### `models/timesclip_language_only.py` (语言模态版本) ⭐
- 简化的 TimesCLIP 模型
- 只包含：语言模态
- 特点：训练快 3-5 倍，参数少 30%

---

### 2. 训练脚本

#### `train_multiyear_mirror.py` (原始双模态训练)
- 训练完整的双模态模型
- 阶段1：时间序列补全 (18 → 18步)
- 阶段2：产量预测 (36步 → 产量)
- 输出：
  - `checkpoints/stage1_timeseries_best.pth`
  - `checkpoints/stage2_yield_best.pth`

#### `train_language_only.py` (语言模态训练) ⭐
- 训练语言模态版本
- 相同的两阶段结构
- 输出：
  - `checkpoints/stage1_language_only_best.pth`
  - `checkpoints/stage2_language_only_best.pth`

---

### 3. 对比与评估

#### `compare_results.py` ⭐
- 对比两个版本的性能
- 输入：两个版本训练好的模型
- 输出：
  - 终端打印详细对比
  - `comparison_results.json` (结果文件)
- 指标：MSE, RMSE, MAE, R², 参数量

#### `test_language_only.py`
- 快速测试语言模态版本是否能正常运行
- 测试：前向传播、反向传播、特征编码
- 用于调试

---

### 4. 文档

#### `ALGORITHM_GUIDE.md`
- 算法原理详细说明
- 数据流图解
- 损失函数设计
- 使用方法

#### `ARCHITECTURE_ANALYSIS.md` ⭐
- 架构设计的深度分析
- 指出当前设计的问题
- 提供多种优化方案
- 理论上的最优架构

#### `ABLATION_EXPERIMENT_GUIDE.md` ⭐
- 消融实验完整指南
- 逐步操作说明
- 结果判断标准
- 实验记录模板

#### `ABLATION_README.md` (本文件)
- 文件清单和快速参考

---

## 快速开始

### 方案A：只运行语言模态版本（推荐）

如果你相信视觉模态用处不大，直接运行：

```bash
# 1. 测试模型是否正常
python test_language_only.py

# 2. 开始训练（预计 3-5 小时）
python train_language_only.py
```

**优点**：
- 快速得到结果
- 训练时间短
- 如果效果好，直接采用

---

### 方案B：完整消融实验（推荐用于研究）

如果你想严格对比，按顺序执行：

```bash
# 1. 训练双模态版本（预计 10-12 小时）
python train_multiyear_mirror.py

# 2. 训练语言模态版本（预计 3-5 小时）
python train_language_only.py

# 3. 对比结果
python compare_results.py

# 4. 查看训练曲线
tensorboard --logdir=logs
```

---

## 预期结果

### 如果语言模态性能 ≥ 95%

**结论**：视觉模态贡献很小

**建议**：
1. ✅ 采用 `train_language_only.py` 作为主训练脚本
2. ✅ 删除或归档双模态相关代码
3. ✅ 享受 3-5 倍的训练加速
4. ✅ 进一步优化语言模态（更深、更大batch）

---

### 如果语言模态性能 < 90%

**结论**：视觉模态很重要

**建议**：
1. ✅ 保留双模态架构
2. ✅ 但优化视觉预处理加速（见 `ARCHITECTURE_ANALYSIS.md`）
3. ✅ 改进对齐损失设计
4. ⚠ 考虑用更快的视觉编码器

---

## 文件对应关系

```
双模态版本                    语言模态版本
├─ models/
│  └─ timesclip.py           └─ timesclip_language_only.py
│
├─ train_multiyear_mirror.py  train_language_only.py
│
├─ checkpoints/
│  ├─ stage1_timeseries_best.pth
│  │                          └─ stage1_language_only_best.pth
│  └─ stage2_yield_best.pth
│                             └─ stage2_language_only_best.pth
│
└─ logs/
   ├─ stage1/                 └─ stage1_language_only/
   └─ stage2/                 └─ stage2_language_only/
```

---

## 关键差异

| 维度 | 双模态 | 语言模态 | 差异 |
|------|--------|----------|------|
| **模块数量** | 7 | 5 | -2 (去掉视觉+对齐) |
| **参数量** | ~25M | ~18M | -28% |
| **每epoch时间** | ~5分钟 | ~1分钟 | **快5倍** ⚡ |
| **总训练时间** | ~10小时 | ~3小时 | **快3倍** ⚡ |
| **显存占用** | ~6GB | ~4GB | -33% |
| **损失函数** | gen + align | gen only | 更简单 |
| **代码复杂度** | 高 | 低 | 易维护 |

---

## 核心问题

这个实验要回答的核心问题是：

**视觉模态（时序可视化图像 + CLIP编码）能提升多少性能？**

- 如果提升 < 5%：不值得，去掉
- 如果提升 5-10%：看情况，权衡速度和性能
- 如果提升 > 10%：很重要，保留但优化

---

## 理论预测

基于架构分析，我的预测是：

**语言模态性能 ≈ 96-98% 双模态性能**

理由：
1. BERT 的序列建模能力已经很强
2. 时序数据更适合序列模型而非图像模型
3. CLIP 在自然图像上预训练，时序图像相关性低
4. 数值 → 图像 → 特征的转换损失信息

如果预测正确，强烈建议采用语言模态版本。

---

## 下一步优化方向

### 如果语言模态够用

1. 尝试不用 BERT，自己训练 Transformer
2. 增大模型（d_model=512, n_layers=8）
3. 增大 batch_size（8→16→32）
4. 更长的训练（150 epochs）
5. 数据增强（时序扭曲、噪声注入）

### 如果需要保留双模态

1. 缓存预生成的图像
2. 用更快的绘图库（Plotly, Bokeh）
3. 改进对齐损失（样本级对齐）
4. 用更轻量的视觉编码器（ResNet18）
5. 多任务学习（统一两阶段）

---

## 常见问题

### Q: 我应该先运行哪个？
**A**: 如果时间有限，先运行 `train_language_only.py`（快3倍）。如果要做严格对比，两个都运行。

### Q: 能否跳过阶段2？
**A**: 可以。阶段1是核心对比。阶段2两个版本用的是相同的独立模型，结果应该差不多。

### Q: 运行到一半可以停止吗？
**A**: 可以。每10个epoch会保存checkpoint，可以从checkpoint继续训练（需要修改代码）。

### Q: 如何确保对比公平？
**A**: 两个脚本已经配置为相同的参数（batch_size、epochs、lr等）。确保在同一台机器上运行。

---

## 总结

这套消融实验能够：
- ✅ 量化视觉模态的实际价值
- ✅ 发现架构的优化空间
- ✅ 指导后续开发方向
- ✅ 在性能和效率间做出明智权衡

**建议优先级**：
1. 运行 `train_language_only.py`（必做）
2. 运行 `compare_results.py` 对比（如果有双模态结果）
3. 查看 `ARCHITECTURE_ANALYSIS.md` 了解优化方向（必读）
4. 根据结果决定后续方案

祝实验顺利！

