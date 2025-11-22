# 实验追踪系统

## 📌 概述

完整的实验管理系统，用于记录、对比和分析深度学习实验结果，便于论文撰写。

### 核心功能

- ✅ **自动记录**：训练脚本自动记录实验配置和结果
- ✅ **快速对比**：一键生成实验对比报告和可视化图表
- ✅ **论文材料**：自动导出LaTeX表格和CSV数据
- ✅ **历史追踪**：完整保存所有实验的配置和结果
- ✅ **性能分析**：自动计算过拟合、改进率等关键指标

---

## 🚀 快速开始（3步）

### 1. 运行训练

```bash
python train_12steps_dual_cached.py
```

训练结束后**自动记录**到`experiment_logs/experiments.csv`

### 2. 查看结果

```bash
python view_experiments.py --top 5
```

或使用批处理脚本（Windows）：

```bash
.\查看实验.bat
```

### 3. 生成论文材料

```bash
python view_experiments.py --compare --export
```

生成：
- `experiment_logs/comparison_report.md` - 详细对比报告
- `experiment_logs/comparison_plots.png` - 6张对比图表
- `experiment_logs/paper_table.tex` - LaTeX表格
- `experiment_logs/paper_table.csv` - Excel表格

---

## 📂 文件说明

### 核心模块

| 文件 | 说明 | 大小 |
|------|------|------|
| `experiment_tracker.py` | 追踪系统核心代码 | ~450行 |
| `view_experiments.py` | 查看和分析脚本 | ~280行 |
| `test_experiment_tracker.py` | 系统测试脚本 | ~230行 |

### 文档

| 文件 | 说明 |
|------|------|
| `EXPERIMENT_PROTOCOL.md` | 实验记录规范和模板（详细） |
| `实验记录使用指南.md` | 使用指南（中文） |
| `README_实验追踪系统.md` | 本文件 |

### 工具

| 文件 | 说明 |
|------|------|
| `查看实验.bat` | Windows快捷菜单 |

---

## 📊 记录内容

每个实验自动记录40+项指标：

### 配置信息
- 数据配置（time_steps, train/val/test划分）
- 模型配置（dropout, 对比学习权重等）
- 训练配置（batch_size, lr, weight_decay）
- 数据增强配置（模式、概率、类型）
- 损失函数配置（focal_gamma, focal_alpha）

### 性能指标
- 训练集最佳性能（train_f1, train_acc）
- 验证集最佳性能（val_f1, val_acc, best_epoch）
- 测试集最终性能（test_f1, test_acc）
- 各类别F1（class0/1/2/3_f1）
- 过拟合指标（overfit_gap, train_val_f1_ratio）

### 其他信息
- 模型参数量（total_params, trainable_params）
- 训练时长（training_time_hours）
- 自定义备注（notes）

---

## 🎯 典型使用场景

### 场景1：对比数据增强效果

```bash
# 运行4个实验
python train_12steps_dual_cached.py  # 修改augmentation_mode依次为:
# 1. none (baseline)
# 2. light
# 3. medium
# 4. heavy

# 对比结果
python view_experiments.py --augmentation
```

输出：
```
数据增强模式对比
                    best_val_f1         overfit_gap    
                           mean    max          mean
augmentation_mode                                      
none                     0.5630 0.5630        0.3720
light                    0.6050 0.6150        0.2800
medium                   0.6500 0.6500        0.1300
heavy                    0.6800 0.6800        0.0400  ← 最佳
```

### 场景2：找出最佳配置

```bash
python view_experiments.py --top 3
```

输出：
```
🥇 排名 1: exp_20251122_150000
   Val F1: 0.6800 | Test F1: 0.6750
   配置: heavy增强, Dropout=0.3, WD=5e-4
```

### 场景3：生成论文表格

```bash
python view_experiments.py --export
```

生成的`paper_table.tex`可直接用于LaTeX：
```latex
\begin{table}
\caption{Experimental Results}
\begin{tabular}{llrrrr}
\toprule
Description & Aug Mode & Dropout & Val F1 & Test F1 & Gap \\
\midrule
Baseline & none & 0.1 & 0.5630 & 0.5500 & 0.3720 \\
Medium Aug & medium & 0.3 & 0.6500 & 0.6450 & 0.1300 \\
Heavy Aug & heavy & 0.3 & 0.6800 & 0.6750 & 0.0400 \\
\bottomrule
\end{tabular}
\end{table}
```

### 场景4：分析改进幅度

```python
import pandas as pd

df = pd.read_csv('experiment_logs/experiments.csv')
baseline = df[df['augmentation_mode'] == 'none'].iloc[0]
best = df.loc[df['best_val_f1'].idxmax()]

val_f1_improve = (best['best_val_f1'] - baseline['best_val_f1']) / baseline['best_val_f1'] * 100
gap_reduction = (baseline['overfit_gap'] - best['overfit_gap']) / baseline['overfit_gap'] * 100

print(f"Val F1 提升: {val_f1_improve:.2f}%")       # +20.8%
print(f"过拟合降低: {gap_reduction:.2f}%")         # -89.2%
```

---

## 📈 可视化图表

运行`python view_experiments.py --compare`生成6张子图：

1. **Val F1对比柱状图**
   - 绿色：过拟合<0.1（优秀）
   - 橙色：过拟合0.1-0.2（可接受）
   - 红色：过拟合>0.2（需改进）

2. **Train vs Val F1散点图**
   - 靠近对角线：泛化好
   - 远离对角线：过拟合严重

3. **数据增强效果对比**
   - 不同增强模式的平均Val F1

4. **Dropout影响分析**
   - Dropout vs Val F1关系

5. **过拟合差距对比**
   - 红色线：警戒值（0.2）
   - 橙色线：警告值（0.1）

6. **各类别F1分布**
   - 识别哪些类别性能较弱

---

## 💡 最佳实践

### 1. 命名规范

```python
# 好的描述
description = "Heavy增强 + Dropout0.3 + 重新划分数据集"

# 不好的描述
description = "测试1"
```

### 2. 标签使用

```python
# 推荐的标签体系
tags = ['baseline']                    # 基线实验
tags = ['augmentation', 'heavy']       # 数据增强实验
tags = ['ablation', 'no_vision']       # 消融实验
tags = ['final', 'paper']              # 最终模型
tags = ['failed', 'lr_too_high']       # 失败实验也要记录
```

### 3. 详细备注

```python
notes = """
改动：
- 数据增强：medium → heavy
- 验证集：445 → 667样本

观察：
- Class 0性能提升最显著（+15%）
- 收敛变慢（epoch 15 → 25）

问题：
- 训练时间增加20%

下一步：
- 尝试only对少数类增强
"""
```

### 4. 定期备份

```bash
# 每5个实验备份一次
cp -r experiment_logs experiment_logs_backup_$(date +%Y%m%d)

# 或使用git
git add experiment_logs/*.csv experiment_logs/*.json
git commit -m "实验记录：测试heavy增强效果"
```

---

## 🧪 验证系统

运行测试脚本验证系统是否正常：

```bash
python test_experiment_tracker.py
```

测试内容：
- ✅ 基本记录功能
- ✅ 多实验记录
- ✅ 对比报告生成
- ✅ 摘要显示

---

## 📝 论文写作工作流

### 第1步：规划实验

参考`EXPERIMENT_PROTOCOL.md`设计实验矩阵：

```
实验组1（Baseline对比）:
- exp_baseline: 无增强
- exp_light: Light增强
- exp_medium: Medium增强
- exp_heavy: Heavy增强

实验组2（消融实验）:
- exp_full: 完整模型
- exp_no_contrast: 移除对比学习
- exp_no_vision: 移除视觉分支
- exp_no_language: 移除语言分支
```

### 第2步：运行实验

```bash
# 依次运行每个实验
python train_12steps_dual_cached.py  # 修改相应配置
```

### 第3步：生成材料

```bash
# 生成对比报告
python view_experiments.py --compare

# 导出表格
python view_experiments.py --export

# 查看数据增强效果
python view_experiments.py --augmentation > augmentation_analysis.txt
```

### 第4步：编写论文

使用生成的材料：

**表格**：直接使用`paper_table.tex`
```latex
\input{experiment_logs/paper_table.tex}
```

**图表**：插入`comparison_plots.png`
```latex
\includegraphics[width=\textwidth]{experiment_logs/comparison_plots.png}
```

**数据**：从`comparison_report.md`复制统计数据
```markdown
数据增强使Val F1从0.563提升至0.680（+20.8%），
同时过拟合差距从0.372降至0.040（-89.2%）
```

---

## 🔧 高级用法

### 自定义查询

```python
import pandas as pd

df = pd.read_csv('experiment_logs/experiments.csv')

# 找出所有Val F1 > 0.65的实验
high_perf = df[df['best_val_f1'] > 0.65]
print(high_perf[['experiment_id', 'best_val_f1', 'augmentation_mode']])

# 找出过拟合最小的实验
best_generalize = df.loc[df['overfit_gap'].idxmin()]
print(f"最佳泛化实验: {best_generalize['experiment_id']}")

# 计算某配置的平均性能
medium_avg = df[df['augmentation_mode'] == 'medium']['best_val_f1'].mean()
print(f"Medium增强平均Val F1: {medium_avg:.4f}")
```

### 批量对比

```python
from experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# 只对比增强相关实验
tracker.compare_experiments(tags=['augmentation'])

# 只对比特定实验
tracker.compare_experiments(exp_ids=['exp_001', 'exp_002', 'exp_003'])
```

### 导出自定义表格

```python
df = pd.read_csv('experiment_logs/experiments.csv')

# 选择论文需要的列
custom_table = df[[
    'description', 
    'augmentation_mode',
    'best_val_f1', 
    'final_test_f1',
    'overfit_gap'
]].sort_values('best_val_f1', ascending=False)

# 导出为LaTeX
print(custom_table.to_latex(index=False, float_format='%.4f'))
```

---

## 🆘 故障排除

### 问题1：训练完成后没有自动记录

**原因**：训练脚本版本不是最新的

**解决**：
```bash
# 检查是否有ExperimentTracker导入
grep "ExperimentTracker" train_12steps_dual_cached.py

# 如果没有，更新脚本
```

### 问题2：experiments.csv不存在

**解决**：
```python
from experiment_tracker import ExperimentTracker
tracker = ExperimentTracker()  # 自动创建
```

### 问题3：对比报告无法生成

**原因**：实验数量太少（<2个）

**解决**：至少运行2个实验后再生成报告

### 问题4：图表中文乱码

**解决**：
```python
# 在experiment_tracker.py中修改字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
```

---

## 📦 依赖

```txt
pandas
matplotlib
seaborn
```

安装：
```bash
pip install pandas matplotlib seaborn
```

---

## 📖 延伸阅读

- `EXPERIMENT_PROTOCOL.md` - 完整的实验记录规范
- `实验记录使用指南.md` - 中文详细教程
- `DATA_AUGMENTATION_GUIDE.md` - 数据增强使用指南

---

## ✅ 检查清单（论文提交前）

- [ ] 所有关键实验已记录
- [ ] Baseline实验完整
- [ ] 消融实验完整
- [ ] 生成了comparison_report.md
- [ ] 导出了paper_table.tex
- [ ] 保存了comparison_plots.png
- [ ] 每个实验都有描述性的notes
- [ ] 备份了experiment_logs目录
- [ ] 验证了数据的准确性

---

## 📧 支持

如遇问题，检查：
1. `test_experiment_tracker.py` 测试是否通过
2. `experiment_logs/experiments.csv` 是否存在
3. Python依赖是否完整

---

**祝实验顺利，论文发表成功！** 🎓🎉

