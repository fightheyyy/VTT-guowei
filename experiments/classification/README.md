# TimesCLIP 分类任务实验

基于TimesCLIP论文方法的时间序列分类任务实验

## 数据集说明

**数据文件**: `data/2018four.csv`

### 数据结构
- **总样本数**: 5557
- **特征数**: 518 = 14个波段 × 37个时间步
- **类别数**: 4类（0, 1, 2, 3）
- **类别分布**:
  - 类别3: 2388样本（43%）
  - 类别2: 1288样本（23%）
  - 类别1: 1210样本（22%）
  - 类别0: 671样本（12%）

### 波段信息
14个遥感波段特征（每个时间步重复）：
- `blue`: 蓝色波段
- `evi`: 增强植被指数
- `gcvi`: 绿色归一化植被指数
- `green`: 绿色波段
- `LST`: 地表温度
- `lswi`: 地表水分指数
- `ndvi`: 归一化植被指数
- `NIR`: 近红外波段
- `ratio`: 比率指数
- `red`: 红色波段
- `RVI`: 比值植被指数
- `SWIR1`: 短波红外1
- `VH`: VH极化
- `VV`: VV极化

### 数据划分方式

**分层随机划分**（保持类别比例一致）：
- **训练集**: 80% × 90% = 72% (4001样本)
- **验证集**: 80% × 10% = 8% (445样本)
- **测试集**: 20% (1111样本)

## 模型架构

### 1. 双模态模型 (TimesCLIP完整版)

```
输入: (batch_size, 37, 14)
    ↓
┌─────────────────────────────────┐
│  预处理                          │
│  - 视觉: 时序→折线图 (224×224)   │
│  - 语言: Patch划分               │
└─────────────────────────────────┘
           ↓
┌──────────────┐    ┌──────────────┐
│ CLIP-Vision  │    │ CLIP-Text    │
│ 编码器       │    │ 编码器        │
│ (冻结)       │    │ (冻结)        │
└──────────────┘    └──────────────┘
      ↓                   ↓
   [CLS_img]          [CLS_text]
      ↓                   ↓
      └──── InfoNCE ──────┘
      ↓                   ↓
┌─────────────────────────────────┐
│  变量选择模块                    │
│  (跨变量注意力)                  │
└─────────────────────────────────┘
           ↓
      [融合特征]
           ↓
┌─────────────────────────────────┐
│  分类头                          │
│  512 → 256 → num_classes         │
└─────────────────────────────────┘
           ↓
       [logits]
```

**特点**:
- 双模态特征融合
- InfoNCE对比学习对齐
- 变量选择模块
- 参数量: ~131M (可训练: ~8M)

### 2. 纯语言模型

```
输入: (batch_size, 37, 14)
    ↓
┌─────────────────────────────────┐
│  Patch预处理                     │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  CLIP-Text编码器 (冻结)          │
└─────────────────────────────────┘
           ↓
      [CLS_text]
           ↓
┌─────────────────────────────────┐
│  分类头                          │
│  512 → 256 → num_classes         │
└─────────────────────────────────┘
           ↓
       [logits]
```

**特点**:
- 仅使用语言模态
- 无对比学习
- 参数量更少
- 训练更快

## 快速开始

### 1. 运行训练脚本（推荐）

```bash
# Windows
run_classification.bat

# 选择:
# [1] 训练双模态模型
# [2] 训练纯语言模型
# [3] 训练两个模型并对比
# [4] 仅对比已有模型结果
```

### 2. 命令行训练

```bash
# 训练双模态模型
python experiments/classification/train_classification_timesclip.py --model_type dual --epochs 100

# 训练纯语言模型
python experiments/classification/train_classification_timesclip.py --model_type language_only --epochs 100

# 对比两个模型
python experiments/classification/compare_classification_models.py
```

### 3. 参数说明

```bash
--csv_path          数据文件路径 (默认: data/2018four.csv)
--model_type        模型类型: dual 或 language_only (默认: dual)
--batch_size        批次大小 (默认: 32)
--epochs            训练轮数 (默认: 100)
--lr                学习率 (默认: 1e-4)
--contrastive_weight 对比学习权重 (默认: 0.1, 仅双模态)
```

## 实验结果

训练完成后，结果保存在:

```
experiments/classification/timesclip/
├── checkpoints/
│   ├── dual_best.pth               # 双模态最佳模型
│   └── language_only_best.pth      # 纯语言最佳模型
├── results/
│   ├── dual_results.json           # 双模态结果
│   ├── dual_confusion_matrix.png   # 双模态混淆矩阵
│   ├── language_only_results.json  # 纯语言结果
│   └── language_only_confusion_matrix.png
├── comparison/
│   └── comparison_YYYYMMDD_HHMMSS.png  # 对比图
└── logs/
    ├── dual_YYYYMMDD_HHMMSS/       # TensorBoard日志
    └── language_only_YYYYMMDD_HHMMSS/
```

### 评估指标

- **准确率 (Accuracy)**: 分类正确的样本比例
- **精确率 (Precision)**: 预测为正的样本中实际为正的比例
- **召回率 (Recall)**: 实际为正的样本中被预测为正的比例
- **F1分数**: 精确率和召回率的调和平均
- **混淆矩阵**: 各类别的分类详情

## 实验设计

### 对比实验

**目的**: 验证视觉模态对分类任务的贡献

**方案**:
1. 训练双模态模型（TimesCLIP完整版）
2. 训练纯语言模型（仅CLIP-Text）
3. 对比两者的性能差异

**预期**:
- 双模态模型应优于单模态
- 视觉模态提供额外的判别信息
- 对比学习有助于特征对齐

### 消融实验（可选）

可以通过修改模型参数进行消融实验：

```python
# 不使用变量选择
model = TimesCLIPClassifier(
    use_variate_selection=False
)

# 不使用对比学习
model = TimesCLIPClassifier(
    use_contrastive=False
)

# 调整对比学习权重
train_timesclip_classifier(
    contrastive_weight=0.0  # 或 0.05, 0.2 等
)
```

## 常见问题

### 1. 类别不平衡

**问题**: 类别0样本最少（671个），可能导致该类别预测不准

**解决方案**:
- 使用分层采样（已实现）
- 使用类别权重损失
- 数据增强（未实现）

### 2. 过拟合

**症状**: 训练准确率高，验证准确率低

**解决方案**:
- 已实现早停（patience=15）
- 已使用Dropout（0.1）
- 已使用权重衰减（1e-4）
- 可增大Dropout或减小模型容量

### 3. 显存不足

**解决方案**:
```bash
# 减小批次大小
python experiments/classification/train_classification_timesclip.py --batch_size 16

# 或使用纯语言模型（显存需求更小）
python experiments/classification/train_classification_timesclip.py --model_type language_only
```

## 与TimesCLIP论文的对应关系

| 论文部分 | 代码实现 |
|---------|---------|
| Vision Branch | `models/vision_module.py` |
| Language Branch | `models/language_module_clip.py` |
| InfoNCE Loss | `models/contrastive_loss.py` |
| Variate Selection | `models/variate_selection_timesclip.py` |
| Feature Fusion | `models/timesclip_classifier.py` |
| Training Strategy | `experiments/classification/train_classification_timesclip.py` |

## 参考资料

- 论文: "Teaching Time Series to See and Speak: Forecasting with Aligned Visual and Textual Perspectives"
- TimesCLIP论文详解: `TimesCLIP论文方法详解.md`
- 代码对齐检查: `代码与论文对齐检查.md`

## 后续工作

- [ ] 数据增强（时序抖动、幅度缩放等）
- [ ] 超参数搜索（网格搜索/贝叶斯优化）
- [ ] 集成学习（多模型融合）
- [ ] 特征可视化（t-SNE/UMAP）
- [ ] 跨年份泛化测试

