# VTT - Variable-length Timeseries Transformer

基于Transformer的农作物产量预测深度学习模型

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 检查GPU
python check_gpu.py
```

### 运行实验

```bash
# 实验1: 产量预测（推荐）⭐⭐⭐⭐⭐
python experiments/yield_prediction/train.py --quick

# 查看结果
tensorboard --logdir=experiments/yield_prediction/logs
```

## 项目结构

```
VTT/
├── data/                          # 数据文件
│   ├── extract2019_*.csv         # 2019年数据
│   ├── extract2020_*.csv         # 2020年数据
│   ├── extract2021_*.csv         # 2021年数据
│   └── extract2022_*.csv         # 2022年数据（测试集）
│
├── experiments/                   # 实验代码
│   ├── yield_prediction/         # 实验1: 产量预测 ⭐⭐⭐⭐⭐
│   ├── completion_comparison/    # 实验2: 补全对比 ⭐⭐⭐⭐
│   ├── variable_length/          # 实验3: 可变长度 ⭐⭐⭐
│   └── two_stage/                # 实验4: 两阶段训练 ⭐⭐⭐
│
├── models/                        # 模型定义
│   ├── timesclip.py              # 双模态模型
│   ├── timesclip_language_only.py# 语言模态模型
│   ├── simple_yield_predictor.py # 简单产量预测
│   ├── language_module.py        # 语言模块
│   ├── vision_module.py          # 视觉模块
│   ├── yield_predictor.py        # 产量预测头
│   └── ...
│
├── docs/                          # 文档
│   ├── EXPERIMENTS_GUIDE.md      # 实验总览
│   ├── YIELD_PREDICTION_GUIDE.md # 产量预测详解
│   ├── guides/                   # 使用指南
│   └── analysis/                 # 分析文档
│
├── archive/                       # 归档
│   ├── old_scripts/              # 旧脚本
│   └── old_docs/                 # 旧文档
│
├── checkpoints/                   # 模型保存
├── logs/                          # 训练日志
├── predictions/                   # 预测结果
│
├── README.md                      # 本文件
├── requirements.txt               # 依赖包
├── run_all_experiments.py        # 运行所有实验
└── reorganize.bat                 # 项目整理脚本
```

## 数据说明

**数据特点**:
- **时间跨度**: 4年（2019-2022）
- **时间步数**: 36步（每步10天 = 360天 = 1年）
- **波段/指标**: 13个（NIR, NDVI, EVI, RVI, SWIR1, blue, red等）
- **使用波段**: 7个主要波段
- **样本数**: 
  - 训练集: 2019-2021年（约1500样本）
  - 测试集: 2022年（约500样本）
- **目标**: 预测农作物产量（连续值回归）

**数据格式**:
```
columns: NIR_00, NIR_01, ..., NIR_35,  # 36个时间步
         RVI_00, RVI_01, ..., RVI_35,
         ...
         y2019, y2020, y2021, y2022     # 4年产量标签
```

## 实验说明

### 实验1: 产量预测 ⭐⭐⭐⭐⭐

**研究问题**: 找到最短的预测天数

**方法**: 端到端回归，直接从时间序列预测产量

**命令**:
```bash
# 快速测试（2小时，测试4个时间长度）
python experiments/yield_prediction/train.py --quick

# 完整实验（8小时，测试所有时间长度）
python experiments/yield_prediction/train.py
```

**输出**: 
- `experiments/yield_prediction/results/` - 各时间长度的结果
- `experiments/yield_prediction/logs/` - 训练曲线
- `experiments/yield_prediction/checkpoints/` - 最佳模型

**关键发现**:
- 最短有效预测天数
- 性能vs时间长度曲线
- 最优模型checkpoint

---

### 实验2: 补全对比 ⭐⭐⭐⭐

**研究问题**: "补全再回归" vs "直接回归"

**方法**: 
- 方法A: 少量数据 → 补全序列 → 回归预测
- 方法B: 少量数据 → 直接回归预测

**命令**:
```bash
python experiments/completion_comparison/train.py
```

**输出**: 两种方法的性能对比

**关键发现**:
- 哪种方法更好
- 补全是否引入噪音
- 端到端vs两阶段的权衡

---

### 实验3: 可变长度 ⭐⭐⭐

**研究问题**: 任意前N月预测剩余月份

**方法**: Transformer Decoder条件生成

**命令**:
```bash
python experiments/variable_length/train.py
```

**输出**: 
- 不同输入长度的补全效果
- 可视化预测曲线

**应用场景**:
- 在线预测（数据逐步到达）
- 灵活预测（任意时间点预测）

---

### 实验4: 两阶段训练 ⭐⭐⭐

**研究问题**: 标准的两阶段训练

**方法**: 
- 阶段1: 预训练时间序列补全
- 阶段2: 微调产量预测

**命令**:
```bash
python experiments/two_stage/train.py
```

**输出**: 
- 两阶段的性能
- 与实验2对比

---

### 运行所有实验

```bash
# 运行所有实验（约24小时）
python run_all_experiments.py

# 查看结果汇总
python run_all_experiments.py --summarize
```

## 评估指标

所有实验使用统一指标：
- **RMSE**: 均方根误差（越低越好）
- **MAE**: 平均绝对误差（越低越好）
- **R²**: 决定系数（0-1，越高越好）
- **MAPE**: 平均绝对百分比误差（越低越好）

## 文档

### 快速文档
- **本README**: 项目总览
- **实验指南**: `docs/EXPERIMENTS_GUIDE.md`
- **产量预测**: `docs/YIELD_PREDICTION_GUIDE.md`

### 详细文档
- **快速开始**: `docs/guides/QUICK_START.md`
- **训练指南**: `docs/guides/TRAINING_GUIDE.md`
- **两阶段训练**: `docs/guides/TWO_STAGE_GUIDE.md`
- **可变长度**: `docs/guides/VARIABLE_LENGTH_GUIDE.md`

### 分析文档
- **算法分析**: `docs/analysis/ALGORITHM_GUIDE.md`
- **架构分析**: `docs/analysis/ARCHITECTURE_ANALYSIS.md`
- **两阶段vs直接**: `docs/analysis/TWO_STAGE_VS_DIRECT_ANALYSIS.md`

## 模型架构

### 语言模态模型（推荐）

```
Time Series (B, T, V)
        ↓
   Patchify & Embed
        ↓
Transformer Encoder (语言模块)
        ↓
   CLS Token Feature
        ↓
  Yield Predictor Head
        ↓
   Yield (B, 1)
```

### 双模态模型

```
Time Series → Patchify → Language Encoder ─┐
              ↓                             ├→ Fusion → Yield
          Plot Image → Vision Encoder ──────┘
```

## 使用示例

```python
from models.simple_yield_predictor import LanguageOnlyYieldPredictor
import torch

# 创建模型
model = LanguageOnlyYieldPredictor(
    time_steps=18,      # 输入18个月（180天）
    n_variates=7,       # 7个波段
    d_model=256         # 隐藏维度
)

# 准备数据
x = torch.randn(4, 18, 7)  # [Batch=4, Time=18, Variates=7]

# 前向推理
yield_pred = model(x)       # [Batch=4, 1]

print(f"预测产量: {yield_pred}")
```

## 查看结果

```bash
# 查看实验结果文件
ls experiments/*/results/

# 查看训练曲线（TensorBoard）
tensorboard --logdir=experiments/yield_prediction/logs

# 查看图表
start experiments/yield_prediction/results/analysis.png   # Windows
open experiments/yield_prediction/results/analysis.png    # Mac/Linux
```

## 主要特点

✅ **4组独立实验** - 系统性研究不同方法  
✅ **4年真实数据** - 2019-2022年遥感数据  
✅ **端到端训练** - 直接优化目标，避免误差传播  
✅ **灵活预测** - 支持不同输入长度  
✅ **完整文档** - 详细的使用和分析文档  
✅ **清晰结构** - experiments/ 独立组织  

## 系统要求

- **Python**: 3.8+
- **PyTorch**: 1.10+
- **CUDA**: 推荐（GPU加速）
- **RAM**: 16GB+
- **磁盘**: 10GB+

## 整理项目

如果文件夹混乱，运行整理脚本：

```bash
# Windows
reorganize.bat

# Linux/Mac
python reorganize_project.py
```

## 常见问题

### Q1: 网络超时错误？
**A**: 模型已缓存，设置离线模式：
```python
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### Q2: GPU内存不足？
**A**: 减小batch_size或time_steps：
```bash
python experiments/yield_prediction/train.py --batch_size 16 --time_steps 12
```

### Q3: 哪个实验最重要？
**A**: 实验1（产量预测）⭐⭐⭐⭐⭐ - 最核心的实验

### Q4: 语言模态vs双模态？
**A**: 语言模态效果更好，推荐使用

## 引用

```bibtex
@software{vtt2024,
  title={VTT: Variable-length Timeseries Transformer for Crop Yield Prediction},
  year={2024},
}
```

## 许可

MIT License

## 联系

如有问题，请提issue或联系项目维护者。

---

**马上开始**: `python experiments/yield_prediction/train.py --quick` 🚀

