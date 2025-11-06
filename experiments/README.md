# VTT 实验快速指南

## 快速开始

### 1. 单个实验运行

```bash
# 实验1: 产量预测（推荐先运行）
python experiments/yield_prediction/train.py --quick      # 快速测试（2小时）
python experiments/yield_prediction/train.py              # 完整实验（8小时）

# 实验2: 补全对比
python experiments/completion_comparison/train.py         # 4小时

# 实验3: 可变长度
python experiments/variable_length/train.py               # 6小时

# 实验4: 两阶段训练
python experiments/two_stage/train.py                     # 4小时
```

### 2. 运行所有实验

```bash
# 交互式运行所有实验
python run_all_experiments.py
```

### 3. 查看结果

```bash
# 查看实验结果
ls experiments/*/results/

# 查看训练曲线
tensorboard --logdir=experiments/yield_prediction/logs
```

## 实验说明

### 实验1: 产量预测 ⭐⭐⭐⭐⭐
**最重要**，找到最短有效预测天数

**输出**: 
- 12个不同天数的性能对比
- 最优天数推荐
- RMSE vs 天数曲线图

### 实验2: 补全对比 ⭐⭐⭐⭐
验证"补全再回归" vs "直接回归"

**输出**:
- 两种方法的性能对比
- 哪种方法更好的结论

### 实验3: 可变长度 ⭐⭐⭐
任意前N月预测剩余月份

**输出**:
- 通用预测模型
- 支持任意输入长度

### 实验4: 两阶段训练 ⭐⭐⭐
标准的两阶段训练流程

**输出**:
- 两阶段模型
- 各阶段性能

## 推荐顺序

**快速验证**（1天）:
1. 实验1（快速模式）
2. 实验2

**完整研究**（3天）:
1. 实验1（完整模式）
2. 实验2
3. 实验4
4. 实验3

**实际应用**（最简单）:
1. 只运行实验1
2. 选择最优天数
3. 训练单一模型

## 文件结构

```
experiments/
├── yield_prediction/          # 实验1
│   ├── train.py
│   ├── data_loader.py
│   ├── checkpoints/
│   ├── results/
│   └── logs/
│
├── completion_comparison/     # 实验2
│   ├── train.py
│   └── results/
│
├── variable_length/           # 实验3
│   ├── train.py
│   └── results/
│
└── two_stage/                 # 实验4
    ├── train.py
    └── results/
```

## 常见问题

**Q: 先运行哪个实验？**
A: 推荐先运行实验1的快速模式，验证数据和环境正常。

**Q: 内存不足怎么办？**
A: 修改train.py中的batch_size，从32改为16或8。

**Q: 可以只运行一个实验吗？**
A: 可以！每个实验都是独立的。

**Q: 结果保存在哪里？**
A: 每个实验的results/文件夹中。

## 详细文档

查看完整文档：`../EXPERIMENTS_README.md`

---

**开始实验**: `python experiments/yield_prediction/train.py --quick`

