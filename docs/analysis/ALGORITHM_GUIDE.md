# TimesCLIP 算法原理与训练方法说明

## 目录
1. [算法整体架构](#算法整体架构)
2. [数据流与模块详解](#数据流与模块详解)
3. [两阶段训练方法](#两阶段训练方法)
4. [损失函数设计](#损失函数设计)
5. [使用方法](#使用方法)

---

## 算法整体架构

TimesCLIP 是一个基于**双模态对比学习**的时间序列预测模型，核心思想是：
- **将时间序列同时转换为视觉和语言两种模态**
- **通过对比学习对齐两种模态的特征表示**
- **利用预训练的视觉和语言模型（CLIP、BERT）的知识**

### 架构图

```
输入时间序列 [Batch, Time_Steps, N_Variates]
       │
       ├─────────────────────┬─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
视觉预处理器          语言预处理器          变量编码器
(可视化为图像)         (切分为Patch)         (原始序列)
       │                     │                     │
       ▼                     ▼                     │
  CLIP视觉编码          BERT文本编码             │
       │                     │                     │
       ▼                     ▼                     │
  CLS_img              CLS_text, Feat_text        H
  [B,V,D]              [B,V,D], [B,V,P+1,D]    [B,V,D]
       │                     │                     │
       └─────────┬───────────┘                     │
                 ▼                                 │
          对比学习对齐                              │
         (InfoNCE Loss)                            │
                                                   │
            Feat_text ──────────┬──────────────────┘
                 │              │
                 │              ▼
                 │      变量选择模块 (v_CLS)
                 │              │
                 └──────┬───────┘
                        ▼
                     生成器
                        │
                        ▼
            输出预测 [B, N_Variates, Pred_Steps]
```

---

## 数据流与模块详解

### 1. 视觉通路（Vision Pathway）

**输入**: `[Batch, Time_Steps, N_Variates]`

#### 1.1 视觉预处理器 (VisualPreprocessor)
```python
# 将每个变量的时间序列绘制成折线图
for variate in range(N_Variates):
    plot(timeseries[:, variate])  # 生成图像
    
# 输出: [Batch, N_Variates, 3, 224, 224]
```

- **作用**: 将数值时间序列转换为视觉图像
- **实现**: 使用 matplotlib 绘制折线图，每个变量一张图
- **特点**: 保留了时序的整体趋势和模式

#### 1.2 视觉模块 (VisionModule)
```python
# 使用预训练 CLIP 视觉编码器
vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

for variate_images in images:
    cls_token = vision_encoder(variate_images)  # 提取CLS token
    
# 输出: CLS_img [Batch, N_Variates, D_Model]
```

- **作用**: 提取图像的高层次语义特征
- **预训练优势**: 利用 CLIP 在大规模图像-文本对上的预训练知识
- **输出**: 每个变量的视觉特征向量

---

### 2. 语言通路（Language Pathway）

**输入**: `[Batch, Time_Steps, N_Variates]`

#### 2.1 语言预处理器 (LanguagePreprocessor)
```python
# 将时间序列切分为 Patch
patches = []
for i in range(0, Time_Steps - patch_length + 1, stride):
    patch = timeseries[i : i+patch_length]
    patches.append(patch)
    
# 输出: [Batch, N_Variates, N_Patches, Patch_Length]
# 例如: Time_Steps=18, patch_length=6, stride=3 → N_Patches=5
```

- **作用**: 将连续序列离散化为 token 序列（类似 NLP 的分词）
- **Patch**: 相当于时序的"词"
- **滑动窗口**: stride < patch_length 允许重叠，保留更多信息

#### 2.2 语言模块 (LanguageModule)
```python
# 使用预训练 BERT 文本编码器
text_encoder = BertModel.from_pretrained("bert-base-uncased")

# 将每个 Patch 投影到 BERT 的词向量空间
patch_embeddings = linear_projection(patches)

# BERT 编码
output = text_encoder(patch_embeddings)

# 输出:
# - CLS_text: [Batch, N_Variates, D_Model]  # 全局特征
# - Feat_text: [Batch, N_Variates, N_Patches+1, D_Model]  # 所有token特征
```

- **作用**: 将时间序列视为"语言"进行编码
- **预训练优势**: 利用 BERT 的序列建模能力
- **双输出**: 既有全局特征（CLS），也有局部特征（所有 Patch）

---

### 3. 多模态对齐 (ContrastiveAlignment)

**输入**: `CLS_img [B, V, D]` 和 `CLS_text [B, V, D]`

```python
# L2 归一化
CLS_img_norm = normalize(CLS_img)
CLS_text_norm = normalize(CLS_text)

# 计算相似度矩阵
similarity = CLS_img_norm @ CLS_text_norm.T / temperature
# shape: [B×V, B×V]

# InfoNCE 损失（对比学习）
# 对角线是正样本对（同一样本同一变量的视觉-语言特征）
# 非对角线是负样本对
loss = CrossEntropy(similarity, diagonal_labels)
```

**损失函数物理意义**:
- 拉近正样本对：同一变量的视觉和语言特征应该相似
- 推远负样本对：不同变量/样本的特征应该不同
- **效果**: 强制两种模态学到一致的语义表示

---

### 4. 变量选择 (VariateSelection)

**输入**: 
- `CLS_text [B, V, D]`: 语言模态的变量特征
- `H [B, V, D]`: 原始时序的变量特征（由 VariateEncoder 编码）

```python
# 计算变量重要性权重
attention_weights = Attention(CLS_text, H)  # [B, V, V]

# 加权融合
v_CLS = weighted_sum(CLS_text, attention_weights)  # [B, V, D]
```

- **作用**: 自动选择重要的变量（如 NDVI、EVI 等）
- **机制**: 变量间的交叉注意力
- **效果**: 不同时间步可能关注不同的变量组合

---

### 5. 生成器 (Generator)

**输入**:
- `Feat_text [B, V, P+1, D]`: 所有 Patch 的特征
- `v_CLS [B, V, D]`: 变量选择后的特征

```python
# 融合特征
combined = Feat_text + v_CLS.unsqueeze(2)  # 广播加法

# Transformer 解码
decoder_output = TransformerDecoder(combined)

# 投影到预测步长
Y_pred = Linear(decoder_output)  # [B, V, Prediction_Steps]
```

- **作用**: 从特征生成未来的时间序列
- **机制**: Transformer 解码器建模时序依赖
- **输出**: 每个变量的未来 Prediction_Steps 个值

---

## 两阶段训练方法

### 阶段1: 时间序列补全（前18步 → 后18步）

**目标**: 训练 TimesCLIP 模型学习时间序列的内在规律

```python
# 数据
输入: 前 18 个时间步  [Batch, 18, 7变量]
输出: 后 18 个时间步  [Batch, 7, 18]

# 损失函数
total_loss = λ_gen × MSE(Y_pred, Y_true) + λ_align × InfoNCE(CLS_img, CLS_text)
           = 1.0 × gen_loss + 0.2 × align_loss
```

**训练配置**:
```python
lookback = 18           # 输入步长
prediction_steps = 18   # 预测步长
batch_size = 8
epochs = 100
lr_vision = 1e-5        # CLIP 投影层学习率（小，因为预训练）
lr_other = 1e-4         # 其他部分学习率
lambda_gen = 1.0        # 生成损失权重
lambda_align = 0.2      # 对齐损失权重
```

**为什么需要这个阶段？**
1. 学习波段时间序列的内在规律和趋势
2. 通过对比学习获得鲁棒的特征表示
3. 为产量预测打好基础

**训练数据**: 2019-2021年（1500样本）  
**测试数据**: 2022年（500样本）

---

### 阶段2: 产量预测（全年36步 → 产量）

**目标**: 基于全年波段数据预测最终产量

```python
# 数据
输入: 全年 36 个时间步  [Batch, 36, 7变量]
输出: 产量值            [Batch, 1]

# 模型: 独立的 YieldPredictor
class YieldPredictor:
    - Input Projection: 7 → 256
    - Position Embedding: 学习位置信息
    - Transformer Encoder: 4层，8头
    - Global Pooling: 汇聚时序信息
    - MLP Head: 256 → 128 → 64 → 1

# 损失函数
loss = MSE(yield_pred, yield_true)
```

**训练配置**:
```python
time_steps = 36         # 全年数据
batch_size = 8
epochs = 30
lr = 1e-4
d_model = 256
n_heads = 8
n_layers = 4
dropout = 0.3
```

**为什么是独立模型？**
- 产量预测需要全年信息，与阶段1的18步补全任务不同
- 独立训练可以针对性地优化产量预测
- Transformer 结构足够强大，可以端到端学习

**评估指标**:
- MSE Loss: 均方误差
- R² Score: 决定系数（越接近1越好）

---

## 损失函数设计

### 1. Gen Loss (生成损失)

```python
gen_loss = MSE(Y_pred, Y_true)
         = mean((Y_pred - Y_true)²)
```

**作用**: 直接优化预测准确度

**你的训练结果**:
- 初始: 0.538 → RMSE ≈ 0.73
- 最终: 0.178 → RMSE ≈ 0.42
- 数据范围: [-7.19, 12.48]，跨度 ≈ 20
- 相对误差: 0.42 / 20 ≈ **2.1%**

**解读**: 模型能够以约2%的相对误差预测波段值，效果不错。

---

### 2. Align Loss (对齐损失)

```python
# InfoNCE 对比学习损失
similarity = normalize(CLS_img) @ normalize(CLS_text).T / temperature
align_loss = -mean(log(exp(similarity[i,i]) / sum(exp(similarity[i,:]))))
```

**作用**: 强制视觉和语言模态学到一致的特征

**你的训练结果**:
- 初始: 4.01 ≈ log(56)
  - 随机初始化，56个候选中找1个正样本的交叉熵
  - 56 = Batch_Size × N_Variates = 8 × 7
- 最终: 1.20 ≈ log(3.3)
  - 相当于在3-4个候选中找正样本
  - 准确率约 30-33%

**解读**: 模型学到了部分对齐，但还有提升空间。理想情况应降到 0.5 以下。

---

### 3. 总损失

```python
total_loss = λ_gen × gen_loss + λ_align × align_loss
           = 1.0 × gen_loss + 0.2 × align_loss
```

**权重设计**:
- `λ_gen = 1.0`: 主要目标是预测准确
- `λ_align = 0.2`: 对齐损失作为正则化，防止过拟合

**为什么 λ_align 较小？**
- gen_loss ∈ [0, 几百]（MSE，未归一化）
- align_loss ∈ [0, 10]（交叉熵，已归一化）
- 量级不同，需要平衡

---

## 使用方法

### 训练

```bash
# 在 1080 Ti 电脑上运行
python train_multiyear_mirror.py
```

**训练过程**:
1. 阶段1自动运行 100 epochs
2. 保存最佳模型到 `checkpoints/stage1_timeseries_best.pth`
3. 阶段2自动运行 30 epochs
4. 保存最佳模型到 `checkpoints/stage2_yield_best.pth`

**日志查看**:
```bash
tensorboard --logdir=logs
```
- 访问 `http://localhost:6006` 查看训练曲线

---

### 推理

#### 时间序列补全
```python
import torch
from models import TimesCLIP

# 加载模型
model = TimesCLIP(
    time_steps=18,
    n_variates=7,
    prediction_steps=18,
    d_model=256
)
model.load_state_dict(torch.load('checkpoints/stage1_timeseries_best.pth'))
model.eval()

# 预测
x = torch.randn(1, 18, 7)  # 前18步
with torch.no_grad():
    y_pred = model(x, return_loss=False)  # [1, 7, 18] 后18步
```

#### 产量预测
```python
from models.yield_predictor import YieldPredictor

# 加载模型
model = YieldPredictor(
    n_variates=7,
    time_steps=36,
    d_model=256
)
model.load_state_dict(torch.load('checkpoints/stage2_yield_best.pth'))
model.eval()

# 预测
x = torch.randn(1, 36, 7)  # 全年36步
with torch.no_grad():
    yield_pred = model(x)  # [1, 1] 产量值
```

---

## 关键参数说明

### 模型参数

| 参数 | 阶段1 | 阶段2 | 说明 |
|------|-------|-------|------|
| `time_steps` | 18 | 36 | 输入时间步长 |
| `prediction_steps` | 18 | - | 预测步长（仅阶段1） |
| `n_variates` | 7 | 7 | 变量数量（波段数） |
| `d_model` | 256 | 256 | 隐藏维度 |
| `patch_length` | 6 | - | Patch 长度 |
| `stride` | 3 | - | Patch 步长 |
| `n_heads` | 8 | 8 | 注意力头数 |
| `n_layers` | - | 4 | Transformer 层数 |

### 训练参数

| 参数 | 阶段1 | 阶段2 | 说明 |
|------|-------|-------|------|
| `batch_size` | 8 | 8 | 批次大小（1080Ti适用） |
| `epochs` | 100 | 30 | 训练轮数 |
| `lr_vision` | 1e-5 | - | CLIP学习率（小，预训练） |
| `lr_other` | 1e-4 | 1e-4 | 其他部分学习率 |
| `lambda_gen` | 1.0 | - | 生成损失权重 |
| `lambda_align` | 0.2 | - | 对齐损失权重 |
| `dropout` | - | 0.3 | Dropout 比例 |

---

## 模型特点与优势

### 1. 双模态学习
- **视觉**: 捕捉整体趋势和模式
- **语言**: 捕捉局部细节和序列依赖
- **协同**: 两种视角互补，特征更鲁棒

### 2. 预训练知识迁移
- **CLIP**: 图像理解能力
- **BERT**: 序列建模能力
- **效果**: 小数据集也能训练好

### 3. 对比学习正则化
- **InfoNCE**: 防止过拟合
- **对齐约束**: 强制学习有意义的特征
- **泛化**: 提升在测试集上的表现

### 4. 两阶段设计
- **解耦**: 时序补全和产量预测分开
- **灵活**: 可单独使用或组合
- **效率**: 针对性优化各任务

---

## 改进建议

### 当前性能
- Gen Loss: 0.178 (RMSE ≈ 0.42) ✓
- Align Loss: 1.20 ⚠ (还有下降空间)
- Val Loss: 0.532 ✓

### 可尝试的改进

1. **继续训练**: 100 → 150 epochs，Align Loss 可能进一步降低

2. **数据增强**: 
   - 时间序列随机裁剪
   - 高斯噪声注入
   - 时间扭曲

3. **调整权重**: 
   - 增大 λ_align 到 0.5，强化对齐

4. **学习率调度**:
   ```python
   scheduler = CosineAnnealingLR(optimizer, T_max=100)
   ```

5. **Early Stopping**:
   - 监控 Val Loss，20 epochs 不降则停止

---

## 常见问题

### Q1: 为什么 Align Loss 初始值是 4.0？
**A**: 随机初始化时，在 56 个候选中找 1 个正样本，交叉熵 ≈ log(56) ≈ 4.0。这是正常的起始值。

### Q2: Gen Loss 和 Align Loss 哪个更重要？
**A**: Gen Loss 直接决定预测准确度，更重要。Align Loss 是正则化，帮助学习更好的特征。

### Q3: 为什么阶段2不用阶段1的模型？
**A**: 当前设计是独立训练。也可以尝试用阶段1提取特征输入阶段2，但需要调整代码。

### Q4: 1080Ti 显存不够怎么办？
**A**: 
- 减小 `batch_size`: 8 → 4
- 减小 `d_model`: 256 → 128
- 使用梯度累积

### Q5: 如何判断模型训练好了？
**A**:
- Val Loss 不再下降（至少10 epochs）
- Align Loss < 0.5
- Gen Loss 对应的 RMSE < 1% 数据范围

---

## 参考文献

1. **CLIP**: Learning Transferable Visual Models From Natural Language Supervision
2. **BERT**: Pre-training of Deep Bidirectional Transformers
3. **InfoNCE**: Representation Learning with Contrastive Predictive Coding
4. **Patching**: A Time Series is Worth 64 Words (PatchTST)

---

## 联系与支持

- 模型代码: `models/timesclip.py`
- 训练脚本: `train_multiyear_mirror.py`
- 数据加载: `data_loader_multiyear.py`
- 检查点目录: `checkpoints/`
- 日志目录: `logs/`

建议配合 TensorBoard 监控训练过程，及时发现问题。

