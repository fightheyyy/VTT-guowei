# VTT 项目整理说明

## 整理前后对比

### 整理前（混乱）

```
VTT/
├── 20+ 训练脚本（train_*.py）散落在根目录
├── 30+ markdown文档散落在根目录
├── 4个CSV数据文件在根目录
├── experiments/ （新建，但未完全整合）
└── 各种测试脚本...
```

### 整理后（清晰）

```
VTT/
├── data/                          # 数据文件（集中）
│   ├── extract2019_*.csv
│   ├── extract2020_*.csv
│   ├── extract2021_*.csv
│   └── extract2022_*.csv
│
├── experiments/                   # 实验代码（4个独立实验）
│   ├── yield_prediction/         # 实验1: 产量预测 ⭐
│   │   ├── train.py
│   │   └── data_loader.py
│   ├── completion_comparison/    # 实验2: 补全对比
│   │   └── train.py
│   ├── variable_length/          # 实验3: 可变长度
│   │   └── train.py
│   └── two_stage/                # 实验4: 两阶段
│       └── train.py
│
├── models/                        # 模型定义（核心代码）
│   ├── timesclip.py
│   ├── timesclip_language_only.py
│   ├── simple_yield_predictor.py
│   └── ...
│
├── docs/                          # 文档（分类整理）
│   ├── EXPERIMENTS_GUIDE.md
│   ├── YIELD_PREDICTION_GUIDE.md
│   ├── guides/                   # 使用指南
│   │   ├── QUICK_START.md
│   │   ├── TRAINING_GUIDE.md
│   │   ├── TWO_STAGE_GUIDE.md
│   │   └── VARIABLE_LENGTH_GUIDE.md
│   └── analysis/                 # 分析文档
│       ├── ALGORITHM_GUIDE.md
│       ├── ARCHITECTURE_ANALYSIS.md
│       └── TWO_STAGE_VS_DIRECT_ANALYSIS.md
│
├── archive/                       # 归档（旧文件）
│   ├── old_scripts/              # 旧训练脚本
│   │   ├── train_language_only.py
│   │   ├── train_multiyear_mirror.py
│   │   ├── compare_results.py
│   │   └── ... (20+ 文件)
│   └── old_docs/                 # 旧文档
│       ├── ABLATION_README.md
│       ├── PROJECT_SUMMARY.md
│       └── ... (10+ 文件)
│
├── checkpoints/                   # 模型保存
├── logs/                          # 训练日志
├── predictions/                   # 预测结果
│
├── README.md                      # 新README（简洁）
├── requirements.txt
├── run_all_experiments.py        # 统一入口
└── reorganize.bat                # 整理脚本
```

## 整理原则

### 1. 清晰分类
- **data/**: 所有数据文件
- **experiments/**: 独立实验，每个一个文件夹
- **models/**: 核心模型代码
- **docs/**: 文档分类（guides + analysis）
- **archive/**: 归档旧文件

### 2. 保留核心
保留在根目录的文件：
- `README.md` - 主README
- `requirements.txt` - 依赖
- `run_all_experiments.py` - 统一入口
- `check_gpu.py` - GPU检查
- 其他通用工具脚本

### 3. 归档历史
归档但不删除：
- 旧的训练脚本（20+ 文件）
- 旧的文档（10+ 文件）
- 方便以后参考

## 核心改进

### 改进1: 实验独立化

**之前**: 20+ 训练脚本散落在根目录
```
train.py
train_multiyear.py
train_two_stage.py
train_language_only.py
train_variable_length.py
train_yield_prediction.py
...
```

**现在**: 4个独立实验，结构清晰
```
experiments/
├── yield_prediction/train.py      # 实验1
├── completion_comparison/train.py # 实验2
├── variable_length/train.py       # 实验3
└── two_stage/train.py             # 实验4
```

### 改进2: 文档分类

**之前**: 30+ markdown散落
```
README.md
QUICK_START.md
TRAINING_GUIDE.md
ALGORITHM_GUIDE.md
ARCHITECTURE_ANALYSIS.md
...（30+ 文件）
```

**现在**: 分类整理
```
docs/
├── EXPERIMENTS_GUIDE.md           # 主文档
├── YIELD_PREDICTION_GUIDE.md      # 主文档
├── guides/                        # 使用类
│   ├── QUICK_START.md
│   └── TRAINING_GUIDE.md
└── analysis/                      # 分析类
    ├── ALGORITHM_GUIDE.md
    └── ARCHITECTURE_ANALYSIS.md
```

### 改进3: 数据集中

**之前**: CSV文件在根目录
```
extract2019_20251010_165007.csv
extract2020_20251010_165007.csv
extract2021_20251010_165007.csv
extract2022_20251010_165007.csv
```

**现在**: 集中到data/
```
data/
├── extract2019_20251010_165007.csv
├── extract2020_20251010_165007.csv
├── extract2021_20251010_165007.csv
└── extract2022_20251010_165007.csv
```

## 使用指南

### 整理步骤

```bash
# 1. 运行整理脚本
reorganize.bat              # Windows
python reorganize_project.py # Linux/Mac

# 2. 检查新README
cat README_NEW.md

# 3. 替换旧README（如果满意）
mv README_NEW.md README.md  # Linux/Mac
move README_NEW.md README.md # Windows

# 4. 测试实验
python experiments/yield_prediction/train.py --quick

# 5. 提交更改
git add .
git commit -m "Reorganize project structure"
```

### 快速开始

```bash
# 1. 查看新README
cat README.md

# 2. 运行核心实验
python experiments/yield_prediction/train.py --quick

# 3. 查看结果
ls experiments/yield_prediction/results/

# 4. 查看训练曲线
tensorboard --logdir=experiments/yield_prediction/logs
```

## 文件映射

### 训练脚本映射

| 旧位置 | 新位置 |
|--------|--------|
| `train_yield_prediction.py` | `experiments/yield_prediction/train.py` |
| `compare_methods.py` | `experiments/completion_comparison/train.py` |
| `train_variable_length.py` | `experiments/variable_length/train.py` |
| `train_two_stage.py` | `experiments/two_stage/train.py` |
| `train_language_only.py` | `archive/old_scripts/` |
| `train_multiyear_mirror.py` | `archive/old_scripts/` |

### 文档映射

| 旧位置 | 新位置 |
|--------|--------|
| `EXPERIMENTS_README.md` | `docs/EXPERIMENTS_GUIDE.md` |
| `YIELD_PREDICTION_README.md` | `docs/YIELD_PREDICTION_GUIDE.md` |
| `QUICK_START.md` | `docs/guides/QUICK_START.md` |
| `TRAINING_GUIDE.md` | `docs/guides/TRAINING_GUIDE.md` |
| `ALGORITHM_GUIDE.md` | `docs/analysis/ALGORITHM_GUIDE.md` |
| `ARCHITECTURE_ANALYSIS.md` | `docs/analysis/ARCHITECTURE_ANALYSIS.md` |

### 数据映射

| 旧位置 | 新位置 |
|--------|--------|
| `extract*.csv` (根目录) | `data/extract*.csv` |

## 注意事项

### ✅ 保留了什么
- 所有核心代码（models/）
- 所有实验代码（experiments/）
- 所有文档（重新组织）
- 所有数据（移动到data/）
- 所有旧文件（移动到archive/）

### ❌ 删除了什么
- 无（所有文件都保留或归档）

### 🔄 需要更新的
如果你的脚本引用了数据路径，需要更新：

```python
# 旧路径
df = pd.read_csv('extract2019_20251010_165007.csv')

# 新路径
df = pd.read_csv('data/extract2019_20251010_165007.csv')
```

## 检查清单

整理完成后，检查：

- [ ] data/ 文件夹包含4个CSV文件
- [ ] experiments/ 包含4个实验文件夹
- [ ] docs/ 包含文档（guides/ 和 analysis/）
- [ ] archive/ 包含旧文件（old_scripts/ 和 old_docs/）
- [ ] README.md 已更新（简洁清晰）
- [ ] 实验可以正常运行
- [ ] 所有路径引用已更新

## 常见问题

### Q1: 旧脚本还能用吗？
**A**: 能，在archive/old_scripts/，但推荐用新的experiments/结构

### Q2: 文档去哪了？
**A**: docs/ 文件夹，分类整理：
- docs/guides/ - 使用指南
- docs/analysis/ - 分析文档

### Q3: 数据文件去哪了？
**A**: data/ 文件夹

### Q4: 会丢失文件吗？
**A**: 不会，所有文件都保留或归档，没有删除

## 总结

### 整理前
- 50+ 文件在根目录
- 难以找到核心文件
- 不清楚从哪开始

### 整理后
- 清晰的目录结构
- 4个独立实验
- 一目了然的README
- 容易上手

---

**开始使用**: `python experiments/yield_prediction/train.py --quick` 🚀

