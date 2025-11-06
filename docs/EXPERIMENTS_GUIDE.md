```
# VTT 实验项目结构

## 项目概览

本项目包含4组独立实验，每组实验在单独的文件夹中，使用4年的遥感数据（2019-2022）进行农作物产量预测研究。

## 数据说明

### 数据文件
- `extract2019_20251010_165007.csv` - 2019年数据
- `extract2020_20251010_165007.csv` - 2020年数据  
- `extract2021_20251010_165007.csv` - 2021年数据
- `extract2022_20251010_165007.csv` - 2022年数据（测试集）

### 数据特征
- **时间步数**: 36步（每步10天 = 360天 ≈ 1年）
- **波段数**: 13个波段/指数（NIR, NDVI, EVI, RVI等）
- **样本数**: 每年~500个地块样本
- **目标**: 预测产量（y2019, y2020, y2021, y2022）

### 训练/测试划分
- **训练集**: 2019-2021年（3年数据）
- **测试集**: 2022年（1年数据）
- **总训练样本**: ~1500个
- **总测试样本**: ~500个

## 实验目录结构

```
VTT/
├── experiments/                    # 实验文件夹
│   ├── yield_prediction/          # 实验1: 产量预测
│   │   ├── train.py              # 训练脚本
│   │   ├── data_loader.py        # 数据加载
│   │   ├── checkpoints/          # 模型保存
│   │   ├── results/              # 实验结果
│   │   └── logs/                 # 训练日志
│   │
│   ├── completion_comparison/     # 实验2: 补全对比
│   │   ├── train.py              # 训练脚本
│   │   ├── results/              # 实验结果
│   │   └── checkpoints/          # 模型保存
│   │
│   ├── variable_length/           # 实验3: 可变长度
│   │   ├── train.py              # 训练脚本
│   │   ├── results/              # 实验结果
│   │   └── checkpoints/          # 模型保存
│   │
│   └── two_stage/                 # 实验4: 两阶段训练
│       ├── train.py              # 训练脚本
│       ├── results/              # 实验结果
│       └── checkpoints/          # 模型保存
│
├── models/                        # 共享模型定义
│   ├── simple_yield_predictor.py # 产量预测模型
│   ├── timesclip.py              # 双模态模型
│   ├── timesclip_language_only.py# 语言模态模型
│   ├── language_module.py        # 语言模块
│   ├── vision_module.py          # 视觉模块
│   ├── preprocessor.py           # 预处理器
│   ├── generator.py              # 生成器
│   ├── alignment.py              # 对齐模块
│   └── variate_selection.py     # 变量选择
│
├── extract*.csv                   # 数据文件
├── EXPERIMENTS_README.md          # 本文件
└── requirements.txt               # 依赖包
```

## 实验说明

### 实验1: 产量预测（最短有效预测天数）

**目标**: 找到最少需要多少天的数据才能准确预测产量

**文件夹**: `experiments/yield_prediction/`

**实验设计**:
- 测试12个不同输入长度: 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36步
- 对应天数: 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360天
- 使用语言模态模型（更快，性能可能更好）
- 端到端训练：时间序列 → 产量

**运行方式**:
```bash
# 快速测试（4个点，约2小时）
python experiments/yield_prediction/train.py --quick

# 完整实验（12个点，约8小时）
python experiments/yield_prediction/train.py

# 指定训练轮数
python experiments/yield_prediction/train.py --epochs 30
```

**输出结果**:
- `experiments/yield_prediction/results/results.json` - 详细数值
- `experiments/yield_prediction/results/analysis.png` - 可视化图表
- `experiments/yield_prediction/checkpoints/*.pth` - 训练好的模型
- `experiments/yield_prediction/logs/` - TensorBoard日志

**预期发现**:
- 最优天数在120-180天（4-6个月）
- R² > 0.75 即可实际应用
- 边际效益递减规律

**应用价值**: 
- 早期预警（60-120天）
- 中期预测（180天）
- 准确预测（240天+）

---

### 实验2: 补全对比（补全 vs 不补全）

**目标**: 对比"先补全序列再回归" vs "直接回归"的性能差异

**文件夹**: `experiments/completion_comparison/`

**实验设计**:
- 测试4个输入长度: 6, 12, 18, 24个月
- 两种方法:
  - 方法1: 直接回归（前N月 → 产量）
  - 方法2: 两阶段（前N月 → 补全36月 → 产量）
- 使用简化的LSTM模型（快速对比）

**运行方式**:
```bash
python experiments/completion_comparison/train.py
```

**输出结果**:
- `experiments/completion_comparison/results/results.json`
- `experiments/completion_comparison/results/comparison.png`

**预期发现**:
- 直接法可能优于两阶段法（避免误差累积）
- 输入越短，直接法优势越大
- 两阶段法可能引入序列预测噪音

**理论依据**:
- 端到端优化更直接
- 避免中间误差传播
- 训练-测试分布一致

---

### 实验3: 可变长度预测

**目标**: 实现任意前N个月预测剩余月份的能力

**文件夹**: `experiments/variable_length/`

**实验设计**:
- 输入: 前N个月（N可以是3-30任意值）
- 输出: 预测剩余月份（36-N个月）
- 使用Transformer Decoder架构
- 训练时随机采样不同输入长度

**运行方式**:
```bash
python experiments/variable_length/train.py
```

**输出结果**:
- `experiments/variable_length/results/`
- 训练好的通用模型（支持任意输入长度）

**应用场景**:
- 数据不完整时仍可预测
- 灵活应对不同时间点
- 动态预测系统

**技术特点**:
- 位置编码处理可变长度
- 长度嵌入告知模型当前输入
- 一个模型适配所有长度

---

### 实验4: 两阶段训练

**目标**: 标准的两阶段训练流程（序列补全 + 产量预测）

**文件夹**: `experiments/two_stage/`

**实验设计**:
- Stage1: 训练时间序列补全（前18月 → 后18月）
- Stage2: 冻结Stage1，训练产量预测（完整36月 → 产量）
- 使用双模态模型或语言模态

**运行方式**:
```bash
python experiments/two_stage/train.py
```

**输出结果**:
- `experiments/two_stage/checkpoints/model.pth`
- `experiments/two_stage/results/results.json`

**训练策略**:
1. 先优化序列重建损失
2. 冻结Stage1参数
3. 只训练产量预测头
4. 避免过拟合

---

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install torch numpy pandas scikit-learn matplotlib seaborn tensorboard transformers

# 确保数据文件存在
ls extract*.csv
```

### 2. 运行实验

**推荐顺序**:

```bash
# 第一步：产量预测（快速测试）
python experiments/yield_prediction/train.py --quick

# 第二步：补全对比
python experiments/completion_comparison/train.py

# 第三步：两阶段训练
python experiments/two_stage/train.py

# 第四步：可变长度（可选，更复杂）
python experiments/variable_length/train.py
```

### 3. 查看结果

```bash
# 查看TensorBoard
tensorboard --logdir=experiments/yield_prediction/logs

# 查看图表
open experiments/yield_prediction/results/analysis.png
open experiments/completion_comparison/results/comparison.png

# 查看数值结果
cat experiments/yield_prediction/results/results.json
```

## 实验对比

| 实验 | 时间 | 难度 | 实用性 | 创新性 |
|------|------|------|--------|--------|
| 实验1: 产量预测 | 8小时 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 实验2: 补全对比 | 4小时 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 实验3: 可变长度 | 6小时 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 实验4: 两阶段 | 4小时 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 推荐实验顺序

### 对于快速验证
1. 实验1（快速模式）- 验证数据和模型可行性
2. 实验2 - 验证端到端 vs 两阶段

### 对于完整研究
1. 实验1（完整模式）- 找到最优输入长度
2. 实验2 - 理论验证
3. 实验4 - 标准方法对比
4. 实验3 - 高级功能（可选）

### 对于实际应用
1. 实验1 - 确定最短有效天数
2. 基于实验1结果训练单一最优模型
3. 部署到生产环境

## 共享模块

### models/ 目录

所有实验共享的模型定义：

- **simple_yield_predictor.py**: 简单产量预测模型
  - `LanguageOnlyYieldPredictor`: 仅语言模态（推荐）
  - `SimpleYieldPredictor`: 双模态版本

- **timesclip.py**: 完整双模态模型
  - 包含视觉和语言模态
  - 对比学习对齐
  - 用于完整的TimesCLIP实现

- **timesclip_language_only.py**: 语言模态模型
  - 去除视觉模块
  - 更快、更简单
  - 性能可能更好

### 使用共享模型

```python
import sys
sys.path.append('../..')  # 回到项目根目录

from models.simple_yield_predictor import LanguageOnlyYieldPredictor

model = LanguageOnlyYieldPredictor(
    time_steps=18,
    n_variates=7,
    d_model=256
)
```

## 评估指标

所有实验使用统一的评估指标：

1. **RMSE** (Root Mean Square Error)
   - 主要指标
   - 越低越好
   - 单位与产量相同

2. **MAE** (Mean Absolute Error)
   - 平均绝对误差
   - 更直观理解

3. **R²** (R-squared)
   - 0-1之间
   - >0.75 可接受
   - >0.85 良好
   - >0.9 优秀

4. **MAPE** (Mean Absolute Percentage Error)
   - 百分比误差
   - 便于跨数据集比较

## 结果解读

### 产量预测（实验1）

**好结果**:
- RMSE < 0.5（取决于产量范围）
- R² > 0.8
- 最优点在120-180天

**一般结果**:
- RMSE 0.5-1.0
- R² 0.6-0.8
- 需要更多数据或更好模型

**差结果**:
- RMSE > 1.0
- R² < 0.6
- 检查数据质量或模型设计

### 补全对比（实验2）

**直接法更好**（预期）:
- 直接法RMSE < 两阶段RMSE
- 提升 > 10%
- 结论：不需要补全

**两阶段更好**（意外）:
- 两阶段RMSE < 直接法RMSE
- 说明完整序列信息很重要
- 需要进一步分析原因

## 故障排查

### CUDA OOM
```bash
# 减小batch size
# 修改train.py中的batch_size参数
```

### 模型不收敛
```bash
# 降低学习率
# 增加训练轮数
# 检查数据归一化
```

### 结果不佳
```bash
# 增加模型容量（d_model）
# 尝试不同波段组合
# 检查数据质量
```

## 引用

如果使用本项目，请引用：

```
VTT: Variable-length Timeseries Transformer for Crop Yield Prediction
[Your Institution]
2024
```

## 许可

MIT License

## 联系

如有问题，请提issue或联系项目维护者。

---

**祝实验顺利！** 🚀

每个实验都是独立的，可以单独运行，也可以按顺序运行进行系统性研究。
```

