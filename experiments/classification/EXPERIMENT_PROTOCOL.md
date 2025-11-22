# 实验记录规范

## 📋 实验编号规则

```
exp_YYYYMMDD_HHMMSS_描述
例如: exp_20251122_143052_medium_aug
```

## 📊 必须记录的信息

### 1. 实验配置（Before Training）

| 类别 | 项目 | 说明 |
|------|------|------|
| **数据** | time_steps | 时间步数（6/12/18...） |
| | train/val/test划分 | 样本数量 |
| | 是否重采样 | 解决类别不平衡 |
| **模型** | dropout | Dropout率 |
| | use_contrastive | 是否对比学习 |
| | contrastive_weight | 对比学习权重 |
| **优化** | lr | 学习率 |
| | weight_decay | L2正则化 |
| | batch_size | 批次大小 |
| **增强** | augmentation_mode | none/light/medium/heavy |
| | ts_aug_prob | 时序增强概率 |
| | img_aug_prob | 图像增强概率 |
| | aug_types | 增强类型列表 |
| **损失** | focal_gamma | Focal Loss γ参数 |
| | focal_alpha | Focal Loss α参数 |

### 2. 训练过程（During Training）

每个Epoch记录：
- Train Loss, Train F1, Train Acc
- Val Loss, Val F1, Val Acc
- 学习率（如有变化）

### 3. 最终结果（After Training）

| 指标 | 说明 |
|------|------|
| **验证集** | best_val_f1, best_val_acc, best_epoch |
| **测试集** | final_test_f1, final_test_acc |
| **类别性能** | class0/1/2/3_f1 |
| **过拟合** | overfit_gap = train_f1 - val_f1 |
| **混淆矩阵** | 测试集混淆矩阵 |
| **训练时间** | 总训练时长（小时） |

---

## 🔬 实验类型标签

用于快速分类和检索：

| 标签 | 含义 | 示例 |
|------|------|------|
| `baseline` | 基线实验 | 无增强的初始配置 |
| `augmentation` | 数据增强实验 | 测试各种增强策略 |
| `regularization` | 正则化实验 | 测试dropout/weight_decay |
| `architecture` | 架构实验 | 改变模型结构 |
| `ablation` | 消融实验 | 去除某个组件 |
| `final` | 最终模型 | 论文中使用的模型 |

---

## 📝 实验日志模板

### 单次实验记录

```markdown
# 实验 exp_20251122_143052

## 目标
测试medium数据增强对过拟合的改善效果

## 假设
添加数据增强可以：
1. 降低Train F1（正常现象）
2. 提升Val F1
3. 缩小Train-Val F1差距

## 配置
- **基于**: exp_20251122_120000 (baseline)
- **改变**: 
  - 数据增强: none → medium
  - Dropout: 0.1 → 0.3
  - Weight Decay: 1e-4 → 5e-4

**详细配置**:
```json
{
  "time_steps": 12,
  "dropout": 0.3,
  "weight_decay": 5e-4,
  "augmentation_mode": "medium",
  "ts_aug_prob": 0.7,
  "img_aug_prob": 0.5,
  "aug_types": ["noise", "scale", "shift"]
}
```

## 结果

### 性能对比

| 指标 | Baseline | 本实验 | 变化 |
|------|----------|--------|------|
| Train F1 | 0.935 | 0.780 | -0.155 ✓ |
| Val F1 | 0.563 | 0.650 | +0.087 ✅ |
| Test F1 | 0.550 | 0.645 | +0.095 ✅ |
| 过拟合差距 | 0.372 | 0.130 | -0.242 ✅ |

### 类别性能

| 类别 | Baseline F1 | 本实验 F1 | 变化 |
|------|-------------|-----------|------|
| Class 0 | 0.45 | 0.55 | +0.10 |
| Class 1 | 0.58 | 0.65 | +0.07 |
| Class 2 | 0.60 | 0.68 | +0.08 |
| Class 3 | 0.62 | 0.70 | +0.08 |

### 训练曲线

- Best Epoch: 25 (baseline: 15)
- 训练时间: 3.0小时 (baseline: 2.5小时)
- 早停未触发 / 已触发于第X轮

## 观察

**正面**:
- ✅ Val F1显著提升（+8.7%）
- ✅ 过拟合大幅缓解（差距从37.2%降至13.0%）
- ✅ 所有类别性能均有提升
- ✅ 测试集泛化良好

**负面**:
- ⚠️ 训练时间略微增加（+20%）
- ⚠️ 收敛变慢（best_epoch: 15→25）

**意外发现**:
- Class 0（少数类）提升最显著（+10%）
- 图像增强似乎对Class 2/3效果更好

## 结论

数据增强策略有效，建议：
1. ✅ 采用medium模式作为默认配置
2. 🔍 可尝试heavy模式进一步改善Class 0
3. 🔍 消融实验：分离时序增强和图像增强的贡献

## 后续计划

- [ ] 实验A: 仅时序增强
- [ ] 实验B: 仅图像增强
- [ ] 实验C: heavy模式
- [ ] 实验D: 重新划分数据集（增大验证集）

## 论文材料

**可用于论文的图表**:
- `comparison_plots.png` - 对比图表
- `confusion_matrix_exp_20251122_143052.png` - 混淆矩阵
- `training_curves_exp_20251122_143052.png` - 训练曲线

**关键数据**:
- 数据增强使Val F1从0.563提升至0.650（+15.5%）
- 过拟合差距从0.372降至0.130（-65.1%）
```

---

## 🔄 对比实验记录

### 主题：数据增强效果验证

| 实验ID | 描述 | 增强模式 | Val F1 | 过拟合差距 | 结论 |
|--------|------|----------|--------|-----------|------|
| exp_001 | Baseline | none | 0.563 | 0.372 | 严重过拟合 |
| exp_002 | Light增强 | light | 0.605 | 0.280 | 轻度改善 |
| exp_003 | Medium增强 | medium | 0.650 | 0.130 | 显著改善 ✅ |
| exp_004 | Heavy增强 | heavy | 0.680 | 0.040 | 最佳泛化 🏆 |

**最佳配置**: Heavy增强（待测试集验证）

---

## 📈 消融实验记录

### 主题：各组件贡献度分析

| 实验ID | 移除组件 | Val F1 | 下降幅度 | 组件重要性 |
|--------|---------|--------|---------|-----------|
| exp_full | 完整模型 | 0.680 | - | - |
| exp_no_contrast | 对比学习 | 0.650 | -0.030 | 中等 |
| exp_no_vision | 视觉分支 | 0.620 | -0.060 | 重要 ⭐⭐ |
| exp_no_language | 语言分支 | 0.600 | -0.080 | 重要 ⭐⭐⭐ |
| exp_no_variate_sel | 变量选择 | 0.665 | -0.015 | 较小 |

**结论**: 语言分支 > 视觉分支 > 对比学习 > 变量选择

---

## 📊 论文撰写清单

### 实验部分需要的内容

- [ ] **数据集描述**
  - [ ] 样本分布表
  - [ ] 类别不平衡统计
  - [ ] 训练/验证/测试划分说明

- [ ] **Baseline性能**
  - [ ] 表格：Baseline各类别F1
  - [ ] 混淆矩阵
  - [ ] 问题分析（过拟合）

- [ ] **改进策略**
  - [ ] 数据增强方法描述
  - [ ] 正则化策略
  - [ ] 超参数选择

- [ ] **消融实验**
  - [ ] 各组件贡献度表格
  - [ ] 可视化对比图

- [ ] **最终性能**
  - [ ] 与Baseline对比表格
  - [ ] 与其他方法对比
  - [ ] 训练曲线图
  - [ ] 混淆矩阵对比

- [ ] **可视化材料**
  - [ ] 数据增强示例图
  - [ ] 训练曲线对比
  - [ ] 类别F1对比柱状图
  - [ ] t-SNE特征可视化

---

## 💡 实验技巧

### 1. 快速对比

使用tracker快速生成对比报告：
```bash
python -c "from experiment_tracker import ExperimentTracker; \
           t = ExperimentTracker(); \
           t.compare_experiments(tags=['augmentation'])"
```

### 2. 筛选最佳

找出Val F1 > 0.65的所有实验：
```python
import pandas as pd
df = pd.read_csv('experiment_logs/experiments.csv')
best = df[df['best_val_f1'] > 0.65].sort_values('best_val_f1', ascending=False)
print(best[['experiment_id', 'description', 'best_val_f1', 'augmentation_mode']])
```

### 3. 生成论文表格

直接从CSV生成LaTeX表格：
```python
df = pd.read_csv('experiment_logs/experiments.csv')
df_paper = df[['description', 'best_val_f1', 'final_test_f1', 'overfit_gap']]
print(df_paper.to_latex(index=False))
```

---

## 📁 文件组织

```
experiments/classification/
├── experiment_logs/              # 实验记录目录
│   ├── experiments.csv          # 主记录表
│   ├── exp_001_detail.json      # 详细配置
│   ├── exp_002_detail.json
│   ├── comparison_report.md     # 对比报告
│   └── comparison_plots.png     # 对比图表
│
├── checkpoints/                  # 模型checkpoints
│   ├── timesclip_12steps_dual_20251122_143052/
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   └── training_log.txt
│
└── paper_materials/              # 论文材料
    ├── figures/
    │   ├── data_augmentation_example.png
    │   ├── training_curves.png
    │   └── confusion_matrices.png
    └── tables/
        ├── performance_comparison.tex
        └── ablation_study.tex
```

---

## 🎯 论文写作建议

### 实验部分结构

```markdown
4. Experiments

4.1 Experimental Setup
  - Dataset description
  - Evaluation metrics
  - Implementation details
  - Baseline configuration

4.2 Baseline Performance
  - Table: Baseline results
  - Analysis: Overfitting issue

4.3 Improvement Strategies
  4.3.1 Data Augmentation
    - Methodology
    - Table: Augmentation modes comparison
    - Figure: Training curves with/without augmentation
  
  4.3.2 Regularization
    - Dropout and Weight Decay
    - Table: Hyperparameter tuning results

4.4 Ablation Study
  - Table: Component contribution analysis
  - Figure: Performance breakdown

4.5 Final Results
  - Table: Comparison with state-of-the-art
  - Figure: Confusion matrix
  - Figure: Per-class F1 comparison
  - Discussion

4.6 Analysis
  - Why data augmentation works
  - Error analysis
  - Case study
```

### 关键论点

1. **问题陈述**: 
   > "初始模型出现严重过拟合（Train F1=0.935 vs Val F1=0.563），限制了泛化性能"

2. **解决方案**:
   > "我们采用多模态数据增强策略，包括时序噪声注入、随机缩放和时间平移"

3. **效果证明**:
   > "数据增强使验证集F1从0.563提升至0.650（+15.5%），同时过拟合差距从0.372降至0.130（-65.1%）"

4. **消融验证**:
   > "消融实验表明语言分支贡献最大（-8.0%），其次是视觉分支（-6.0%）"

