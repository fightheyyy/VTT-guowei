# VTT 文件清单

## 核心文件（根目录）

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `README.md` | 项目总览和快速开始 | ⭐⭐⭐⭐⭐ |
| `requirements.txt` | Python依赖包 | ⭐⭐⭐⭐⭐ |
| `run_all_experiments.py` | 运行所有实验的统一入口 | ⭐⭐⭐⭐⭐ |
| `check_gpu.py` | 检查GPU是否可用 | ⭐⭐⭐⭐ |
| `reorganize.bat` | 项目整理脚本 | ⭐⭐⭐ |
| `PROJECT_ORGANIZATION.md` | 整理说明文档 | ⭐⭐⭐ |
| `FILE_INVENTORY.md` | 本文件清单 | ⭐⭐⭐ |

## 工具脚本（根目录）

| 文件 | 用途 | 使用频率 |
|------|------|----------|
| `data_loader.py` | 基础数据加载器 | 高 |
| `data_loader_multiyear.py` | 多年数据加载器 | 高 |
| `data_loader_with_yield.py` | 带产量标签的数据加载器 | 高 |
| `test_data_loading.py` | 测试数据加载 | 低 |
| `quick_test.py` | 快速测试脚本 | 中 |
| `inference.py` | 推理脚本 | 中 |
| `predict_2025.py` | 2025年预测 | 低 |
| `predict_flexible.py` | 灵活预测 | 低 |
| `example_usage.py` | 使用示例 | 低 |
| `config_example.py` | 配置示例 | 低 |

## experiments/ 实验代码

### 实验1: yield_prediction/ ⭐⭐⭐⭐⭐
**目标**: 找到最短的有效预测天数

```
experiments/yield_prediction/
├── train.py            # 训练脚本
├── data_loader.py      # 数据加载器
├── results/            # 结果保存
├── checkpoints/        # 模型保存
└── logs/               # 训练日志
```

**使用**:
```bash
python experiments/yield_prediction/train.py --quick  # 快速测试
python experiments/yield_prediction/train.py          # 完整实验
```

### 实验2: completion_comparison/ ⭐⭐⭐⭐
**目标**: 对比"补全再回归" vs "直接回归"

```
experiments/completion_comparison/
└── train.py            # 训练和对比脚本
```

**使用**:
```bash
python experiments/completion_comparison/train.py
```

### 实验3: variable_length/ ⭐⭐⭐
**目标**: 任意前N月预测剩余月份

```
experiments/variable_length/
└── train.py            # 训练脚本
```

**使用**:
```bash
python experiments/variable_length/train.py
```

### 实验4: two_stage/ ⭐⭐⭐
**目标**: 标准的两阶段训练

```
experiments/two_stage/
└── train.py            # 两阶段训练脚本
```

**使用**:
```bash
python experiments/two_stage/train.py
```

## models/ 模型定义

| 文件 | 模型 | 用途 |
|------|------|------|
| `timesclip.py` | TimesCLIP | 双模态模型（视觉+语言） |
| `timesclip_language_only.py` | TimesCLIPLanguageOnly | 语言模态模型（推荐）⭐ |
| `simple_yield_predictor.py` | SimpleYieldPredictor | 简单产量预测模型 |
| `language_module.py` | LanguageModule | 语言模块（Transformer） |
| `vision_module.py` | VisionModule | 视觉模块（CLIP） |
| `yield_predictor.py` | YieldPredictor | 产量预测头 |
| `variate_selection.py` | VariateSelection | 变量选择模块 |
| `generator.py` | Generator | 序列生成器 |
| `alignment.py` | ContrastiveAlignment | 对比学习对齐 |
| `preprocessor.py` | Preprocessor | 数据预处理 |

## data/ 数据文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `extract2019_20251010_165007.csv` | 2019年数据 | 训练集 |
| `extract2020_20251010_165007.csv` | 2020年数据 | 训练集 |
| `extract2021_20251010_165007.csv` | 2021年数据 | 训练集 |
| `extract2022_20251010_165007.csv` | 2022年数据 | 测试集 |

**数据格式**:
- 13个波段/指标，每个36个时间步
- 4个产量标签（y2019, y2020, y2021, y2022）
- 约500行（样本）

## docs/ 文档

### 主文档
| 文件 | 内容 |
|------|------|
| `EXPERIMENTS_GUIDE.md` | 实验总览 ⭐⭐⭐⭐⭐ |
| `YIELD_PREDICTION_GUIDE.md` | 产量预测详解 ⭐⭐⭐⭐⭐ |

### docs/guides/ 使用指南
| 文件 | 内容 |
|------|------|
| `QUICK_START.md` | 快速开始 |
| `START_HERE.md` | 从这里开始 |
| `TRAINING_GUIDE.md` | 训练指南 |
| `RUN_EXPERIMENT_GUIDE.md` | 实验运行指南 |
| `TWO_STAGE_GUIDE.md` | 两阶段训练指南 |
| `FLEXIBLE_CONFIG_GUIDE.md` | 灵活配置指南 |
| `VARIABLE_LENGTH_GUIDE.md` | 可变长度指南 |

### docs/analysis/ 分析文档
| 文件 | 内容 |
|------|------|
| `ALGORITHM_GUIDE.md` | 算法分析 |
| `ARCHITECTURE_ANALYSIS.md` | 架构分析 |
| `TWO_STAGE_VS_DIRECT_ANALYSIS.md` | 两阶段vs直接回归分析 |
| `EXPERIMENT_DESIGN.md` | 实验设计 |
| `INPUT_MONTHS_COMPARISON.md` | 输入月份对比 |

## archive/ 归档

### archive/old_scripts/ 旧脚本
保留但不再使用的训练脚本：
- `train_language_only.py`
- `train_multiyear_mirror.py`
- `run_ablation_experiment.py`
- `compare_results.py`
- `compare_methods.py`
- ... (20+ 文件)

### archive/old_docs/ 旧文档
保留但不再维护的文档：
- `ABLATION_README.md`
- `PROJECT_SUMMARY.md`
- `PROJECT_COMPLETE.md`
- ... (10+ 文件)

## 其他目录

| 目录 | 内容 | 说明 |
|------|------|------|
| `checkpoints/` | 模型checkpoint | `.gitignore`已忽略`.pth`文件 |
| `logs/` | TensorBoard日志 | `.gitignore`已忽略 |
| `predictions/` | 预测结果图 | `.gitignore`已忽略 |
| `__pycache__/` | Python缓存 | `.gitignore`已忽略 |
| `.git/` | Git仓库 | 版本控制 |

## 文件统计

### 按类型统计
| 类型 | 数量 | 位置 |
|------|------|------|
| Python脚本 | 30+ | 根目录 + models/ + experiments/ |
| Markdown文档 | 30+ | docs/ + archive/ |
| 数据文件 | 4 | data/ |
| 模型定义 | 10 | models/ |
| 实验脚本 | 4 | experiments/ |
| 归档文件 | 30+ | archive/ |

### 按重要性统计
| 重要性 | 文件 |
|--------|------|
| ⭐⭐⭐⭐⭐ 核心 | README, run_all_experiments.py, experiments/ |
| ⭐⭐⭐⭐ 重要 | models/, docs/主文档, data/ |
| ⭐⭐⭐ 有用 | 工具脚本, docs/guides/ |
| ⭐⭐ 参考 | docs/analysis/, archive/ |
| ⭐ 历史 | archive/old_* |

## 快速导航

### 我想...

**开始实验** → `README.md` → `python experiments/yield_prediction/train.py --quick`

**了解算法** → `docs/analysis/ALGORITHM_GUIDE.md`

**查看架构** → `docs/analysis/ARCHITECTURE_ANALYSIS.md`

**训练模型** → `docs/guides/TRAINING_GUIDE.md`

**查看实验设计** → `docs/EXPERIMENTS_GUIDE.md`

**找旧脚本** → `archive/old_scripts/`

**找旧文档** → `archive/old_docs/`

**查看数据** → `data/*.csv`

**查看模型** → `models/*.py`

## 常用命令

```bash
# 快速开始
python experiments/yield_prediction/train.py --quick

# 查看实验结果
ls experiments/*/results/

# 查看训练曲线
tensorboard --logdir=experiments/yield_prediction/logs

# 检查GPU
python check_gpu.py

# 测试数据加载
python test_data_loading.py

# 快速测试模型
python quick_test.py

# 运行所有实验
python run_all_experiments.py
```

## 文件依赖关系

```
实验脚本 (experiments/*/train.py)
    ↓ 依赖
模型定义 (models/*.py)
    ↓ 依赖
数据加载 (data_loader*.py)
    ↓ 依赖
数据文件 (data/*.csv)
```

## 总结

### 核心三要素
1. **实验代码**: `experiments/` - 4个独立实验
2. **模型定义**: `models/` - 10个模型模块
3. **数据文件**: `data/` - 4年数据

### 推荐路径
1. 阅读 `README.md`
2. 运行 `python experiments/yield_prediction/train.py --quick`
3. 查看结果 `experiments/yield_prediction/results/`
4. 深入了解 `docs/EXPERIMENTS_GUIDE.md`

---

**马上开始**: `python experiments/yield_prediction/train.py --quick` 🚀

