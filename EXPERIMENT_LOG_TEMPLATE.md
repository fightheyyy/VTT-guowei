# 实验日志

## 实验记录表

### 实验一：时间步数敏感性分析

#### 双模态模型（SimpleYieldPredictor）

| 实验编号 | 日期 | 步数 | 天数 | RMSE | MAE | R² | MAPE | 训练时长 | Checkpoint | 备注 |
|---------|------|------|------|------|-----|----|----|---------|-----------|------|
| Exp1A-6  | - | 6  | 60  | - | - | - | - | - | - | - |
| Exp1A-12 | - | 12 | 120 | - | - | - | - | - | - | - |
| Exp1A-18 | - | 18 | 180 | - | - | - | - | - | - | - |
| Exp1A-24 | - | 24 | 240 | - | - | - | - | - | - | - |
| Exp1A-30 | - | 30 | 300 | - | - | - | - | - | - | - |
| Exp1A-36 | - | 36 | 360 | - | - | - | - | - | - | - |

#### 纯语言模态（LanguageOnlyYieldPredictor）

| 实验编号 | 日期 | 步数 | 天数 | RMSE | MAE | R² | MAPE | 训练时长 | Checkpoint | 备注 |
|---------|------|------|------|------|-----|----|----|---------|-----------|------|
| Exp1B-6  | - | 6  | 60  | - | - | - | - | - | - | - |
| Exp1B-12 | - | 12 | 120 | - | - | - | - | - | - | - |
| Exp1B-18 | - | 18 | 180 | - | - | - | - | - | - | - |
| Exp1B-24 | - | 24 | 240 | - | - | - | - | - | - | - |
| Exp1B-30 | - | 30 | 300 | - | - | - | - | - | - | - |
| Exp1B-36 | - | 36 | 360 | - | - | - | - | - | - | - |

---

## 详细实验记录

### Exp1A-6: 双模态 6步（60天）

```
==================================================
实验编号: Exp1A-6
日期: YYYY-MM-DD HH:MM
模型: SimpleYieldPredictor (dual-modal)
--------------------------------------------------
数据配置:
  - 输入步数: 6 steps (60 days)
  - 训练集: 2019-2021 (1500 samples)
  - 验证集: 从训练集抽取20% (300 samples)
  - 测试集: 2022 (500 samples)
  - 波段数: 7
--------------------------------------------------
模型配置:
  - d_model: 256
  - patch_length: 6
  - stride: 3
  - use_vision: True
  - use_language: True
  - clip_model: openai/clip-vit-base-patch16
--------------------------------------------------
训练配置:
  - batch_size: 32
  - learning_rate: 1e-4
  - epochs: 100
  - optimizer: AdamW
  - scheduler: ReduceLROnPlateau
  - early_stopping_patience: 15
--------------------------------------------------
训练过程:
  - 最佳epoch: ?
  - 最佳验证loss: ?
  - 是否早停: ?
  - 训练时长: ?
--------------------------------------------------
测试结果:
  - Test RMSE: ?
  - Test MAE: ?
  - Test R²: ?
  - Test MAPE: ?
--------------------------------------------------
文件路径:
  - Checkpoint: experiments/yield_prediction/checkpoints/dual_6steps_best.pth
  - Results: experiments/yield_prediction/results/comparison_results.json
  - Predictions: experiments/yield_prediction/results/dual_6steps_predictions.png
--------------------------------------------------
观察与分析:
  - 
  - 
  - 
--------------------------------------------------
问题记录:
  - 
  - 
==================================================
```

### Exp1B-6: 纯语言 6步（60天）

```
==================================================
实验编号: Exp1B-6
日期: YYYY-MM-DD HH:MM
模型: LanguageOnlyYieldPredictor
--------------------------------------------------
数据配置:
  - 输入步数: 6 steps (60 days)
  - 训练集: 2019-2021 (1500 samples)
  - 验证集: 从训练集抽取20% (300 samples)
  - 测试集: 2022 (500 samples)
  - 波段数: 7
--------------------------------------------------
模型配置:
  - d_model: 256
  - patch_length: 6
  - stride: 3
  - use_vision: False
  - use_language: True
--------------------------------------------------
训练配置:
  - batch_size: 32
  - learning_rate: 1e-4
  - epochs: 100
  - optimizer: AdamW
  - scheduler: ReduceLROnPlateau
  - early_stopping_patience: 15
--------------------------------------------------
训练过程:
  - 最佳epoch: ?
  - 最佳验证loss: ?
  - 是否早停: ?
  - 训练时长: ?
--------------------------------------------------
测试结果:
  - Test RMSE: ?
  - Test MAE: ?
  - Test R²: ?
  - Test MAPE: ?
--------------------------------------------------
文件路径:
  - Checkpoint: experiments/yield_prediction/checkpoints/language_6steps_best.pth
  - Results: experiments/yield_prediction/results/comparison_results.json
  - Predictions: experiments/yield_prediction/results/language_6steps_predictions.png
--------------------------------------------------
观察与分析:
  - 
  - 
  - 
--------------------------------------------------
问题记录:
  - 
  - 
==================================================
```

---

## 实验对比分析

### 双模态 vs 纯语言模态

**性能对比**：

| 步数 | 双模态RMSE | 语言RMSE | 差异 | 获胜者 | 显著性 |
|------|-----------|----------|------|--------|--------|
| 6    | ?         | ?        | ?    | ?      | ?      |
| 12   | ?         | ?        | ?    | ?      | ?      |
| 18   | ?         | ?        | ?    | ?      | ?      |
| 24   | ?         | ?        | ?    | ?      | ?      |
| 30   | ?         | ?        | ?    | ?      | ?      |
| 36   | ?         | ?        | ?    | ?      | ?      |

**分析**：

1. **整体趋势**：
   - 

2. **最优配置**：
   - 

3. **模态贡献**：
   - 

4. **时间敏感性**：
   - 

---

## 结论与下一步

### 主要发现

1. **最短有效时间**：
   - 

2. **最优模型**：
   - 

3. **性能瓶颈**：
   - 

### 下一步计划

- [ ] 
- [ ] 
- [ ] 

---

## 附录

### GPU使用情况
```
GPU型号: 
显存: 
利用率: 
```

### 环境信息
```
Python版本: 
PyTorch版本: 
CUDA版本: 
```

### 其他备注
```

```

