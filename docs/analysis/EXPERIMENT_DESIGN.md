# 完备实验设计方案

## 实验目标

系统性评估序列补全对回归任务的影响

## 实验架构

```
实验A：补全策略对比
├── A1: 直接回归（无补全）
├── A2: 全补全回归（补全到36月）
└── A3: 部分补全回归（补全到不同长度）

实验B：补全长度影响
├── B1: 不同目标长度（18/24/30/36月）
├── B2: 补全质量分析
└── B3: 信息增益测量

实验C：消融实验
├── C1: 模型容量影响
├── C2: 数据量影响
└── C3: 输入长度影响
```

## 实验A：补全 vs 不补全

### A1：直接回归（基线）

**方法**：
```
输入: 前N个月 [Batch, N, 7]
        ↓
    Encoder (LSTM/Transformer)
        ↓
    Regression Head
        ↓
输出: 产量 [Batch, 1]
```

**变量**：
- 输入长度 N ∈ {3, 6, 9, 12, 15, 18, 21, 24}
- 模型容量 D ∈ {64, 128, 256, 512}

### A2：全补全回归

**方法**：
```
输入: 前N个月 [Batch, N, 7]
        ↓
    Stage1: Sequence Completion
        ↓
    完整36个月 [Batch, 36, 7]
        ↓
    Stage2: Regression
        ↓
输出: 产量 [Batch, 1]
```

**关键点**：
- Stage1和Stage2分别训练
- Stage1目标：MSE on 序列重建
- Stage2目标：MSE on 产量预测

### A3：部分补全回归

**方法**：
```
输入: 前N个月
        ↓
    补全到M个月 (N < M < 36)
        ↓
    Regression
        ↓
输出: 产量
```

**实验矩阵**：

| 输入长度N | 补全到M | 补全的月数 |
|----------|---------|-----------|
| 6 | 12 | +6 |
| 6 | 18 | +12 |
| 6 | 24 | +18 |
| 6 | 36 | +30 |
| 12 | 18 | +6 |
| 12 | 24 | +12 |
| 12 | 36 | +24 |

## 实验B：补全长度影响分析

### B1：补全长度扫描

**目标**：找到最优补全长度

**实验设置**：
- 固定输入长度：12个月
- 扫描补全目标：14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36月
- 测量每个补全长度下的回归性能

**假设**：
```
性能曲线可能呈现：
  
  性能
   ↑
   |     ╱‾‾‾╲
   |    ╱     ╲___
   |___╱           ╲___
   +----------------------→ 补全长度
   12  18  24  30  36
   
可能存在最优点（不是越长越好）
```

### B2：补全质量分析

**测量指标**：
1. 序列重建误差（RMSE）
2. 序列相关性（Pearson r）
3. 关键特征保留度
4. 时间对齐误差

**方法**：
```python
# 对于测试集，我们有真实的完整36月数据
真实序列: X_true[36, 7]
预测序列: X_pred[36, 7]

质量指标:
- RMSE_seq = sqrt(mean((X_pred - X_true)²))
- Corr_seq = correlation(X_pred, X_true)
- DTW_distance = dynamic_time_warping(X_pred, X_true)
```

### B3：信息增益测量

**目标**：量化补全带来的信息增益

**方法**：
```python
# 1. 直接回归的不确定性
pred_direct, std_direct = model_direct(X_input)

# 2. 补全后回归的不确定性
X_completed = model_stage1(X_input)
pred_complete, std_complete = model_stage2(X_completed)

# 信息增益
IG = log(std_direct) - log(std_complete)

# 如果IG > 0：补全增加了信息
# 如果IG < 0：补全引入了噪音
```

## 实验C：消融实验

### C1：模型容量影响

**问题**：性能差异是否来自模型容量不同？

**控制实验**：

| 方法 | Stage1参数 | Stage2参数 | 总参数 |
|------|-----------|-----------|--------|
| 直接法-小 | 0 | 100K | 100K |
| 直接法-中 | 0 | 500K | 500K |
| 直接法-大 | 0 | 1M | 1M |
| 两阶段-平衡 | 500K | 500K | 1M |
| 两阶段-Stage1重 | 800K | 200K | 1M |
| 两阶段-Stage2重 | 200K | 800K | 1M |

**保证**：同一行总参数相同，公平比较。

### C2：数据量影响

**问题**：数据量如何影响两种方法？

**实验设置**：
```python
训练集大小 = [100, 300, 500, 1000, 1500] 样本

for size in sizes:
    train_subset = random_sample(train_data, size)
    
    # 训练两种方法
    model_direct = train(train_subset)
    model_twostage = train(train_subset)
    
    # 评估
    results[size] = evaluate(test_data)
```

**预期**：
- 数据少时：直接法可能更好（简单模型）
- 数据多时：两阶段可能改善（有容量学习复杂模式）

### C3：输入长度影响

**完整矩阵实验**：

| 输入(月) | 方法 | RMSE | MAE | R² | 训练时间 |
|---------|------|------|-----|----|----|
| 3 | 直接 | ? | ? | ? | ? |
| 3 | 补全→18 | ? | ? | ? | ? |
| 3 | 补全→36 | ? | ? | ? | ? |
| 6 | 直接 | ? | ? | ? | ? |
| 6 | 补全→18 | ? | ? | ? | ? |
| 6 | 补全→36 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |
| 24 | 直接 | ? | ? | ? | ? |
| 24 | 补全→36 | ? | ? | ? | ? |

共计 **8种输入长度 × 3-5种方法 = 30+次实验**

## 评估指标

### 主要指标

1. **回归性能**
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - R² (决定系数)
   - MAPE (Mean Absolute Percentage Error)

2. **补全质量**（仅两阶段）
   - Sequence RMSE
   - Temporal Correlation
   - DTW Distance

3. **效率指标**
   - 训练时间
   - 推理时间
   - 模型参数量
   - 内存占用

### 统计检验

**配对t检验**：
```python
# 对于同一个测试样本
errors_direct = |y_true - y_pred_direct|
errors_twostage = |y_true - y_pred_twostage|

# 配对t检验
t_stat, p_value = paired_t_test(errors_direct, errors_twostage)

if p_value < 0.05:
    print("差异显著")
```

**Bootstrap置信区间**：
```python
bootstrap_samples = 1000

for i in range(bootstrap_samples):
    sample_idx = random_sample_with_replacement(test_indices)
    rmse_bootstrap[i] = compute_rmse(predictions[sample_idx])

CI_95 = percentile(rmse_bootstrap, [2.5, 97.5])
```

## 可视化分析

### 1. 性能对比图

```
RMSE vs 输入长度

RMSE ↑
  2.0|
     |    ●---● 两阶段
  1.5|     ╲   ╲
     |      ╲   ╲___
  1.0|   ■   ■---■---■ 直接法
     |    ╲___________
  0.5|
     +------------------→ 输入长度
      3  6  12  18  24
```

### 2. 误差分布图

```python
import seaborn as sns

# 小提琴图对比误差分布
plt.subplot(1, 2, 1)
sns.violinplot(data=[errors_direct, errors_twostage])
plt.title('误差分布对比')

# QQ图检验正态性
plt.subplot(1, 2, 2)
scipy.stats.probplot(errors_direct - errors_twostage, plot=plt)
```

### 3. 补全质量热力图

```
变量 ↓  时间步 →
        1  6  12 18 24 30 36
NIR    [■][■][□][□][□][□][□]
NDVI   [■][■][■][□][□][□][□]
EVI    [■][■][□][□][□][□][□]
...

■ = 高质量补全 (RMSE < 0.5)
□ = 低质量补全 (RMSE > 0.5)
```

### 4. 特征重要性分析

```python
# 使用SHAP分析特征重要性
import shap

# 对直接法
explainer_direct = shap.Explainer(model_direct)
shap_values_direct = explainer_direct(X_test)

# 对两阶段法（分别分析两个阶段）
explainer_stage1 = shap.Explainer(model_stage1)
explainer_stage2 = shap.Explainer(model_stage2)

# 可视化：哪些时间步、哪些变量最重要？
```

## 实验流程

### 阶段1：基础对比（1-2天）

```bash
# 快速验证直觉
python experiment_phase1_basic.py \
    --input_lengths 6,12,18 \
    --methods direct,complete \
    --epochs 30
```

**输出**：初步性能对比

### 阶段2：完整实验（3-5天）

```bash
# 完整实验矩阵
python experiment_phase2_full.py \
    --config experiment_config.yaml \
    --n_runs 5 \
    --epochs 50
```

**输出**：
- `results/performance_matrix.csv`
- `results/statistical_tests.csv`
- `figures/comparison_plots/`

### 阶段3：深度分析（2-3天）

```bash
# 消融实验
python experiment_phase3_ablation.py \
    --ablation_type model_capacity,data_size \
    --epochs 50

# 补全质量分析
python analyze_completion_quality.py \
    --checkpoint results/best_models/

# 特征重要性
python analyze_feature_importance.py \
    --method shap
```

## 预期结果

### 假设1：直接法整体更优

**如果成立**：
- 直接法在多数输入长度下RMSE更低
- 差异在输入长度<12月时最显著
- 补全引入的噪音>信息增益

**推论**：
- 端到端学习更有效
- 不需要序列补全这个中间步骤

### 假设2：存在最优补全长度

**如果成立**：
- 补全到某个长度M*时性能最佳
- M* ≠ 36（不是补全越多越好）
- M*可能在18-24之间

**推论**：
- 过度补全引入噪音
- 适度补全可能有帮助

### 假设3：数据量是关键因素

**如果成立**：
- 数据<500时：直接法显著更好
- 数据>2000时：两阶段可能反超
- 交叉点在1000-1500样本

**推论**：
- 当前数据量（1500）处于临界区
- 建议仍然使用直接法（更稳健）

## 实验检查清单

### 实验前

- [ ] 数据划分固定（确保可复现）
- [ ] 随机种子固定
- [ ] 超参数网格搜索完成
- [ ] 基线模型训练完成
- [ ] GPU/计算资源充足

### 实验中

- [ ] 实时监控训练曲线
- [ ] 记录每次实验的配置
- [ ] 保存中间结果
- [ ] 定期备份模型
- [ ] 记录异常情况

### 实验后

- [ ] 统计显著性检验
- [ ] 绘制对比图表
- [ ] 编写实验报告
- [ ] 代码和结果归档
- [ ] 最佳模型打包

## 时间预算

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| 准备 | 数据处理、基线训练 | 1天 |
| 阶段1 | 基础对比实验 | 2天 |
| 阶段2 | 完整矩阵实验 | 5天 |
| 阶段3 | 消融和深度分析 | 3天 |
| 分析 | 数据分析、可视化 | 2天 |
| 总结 | 撰写报告 | 1天 |
| **总计** | | **14天** |

## 交付物

1. **实验报告**（Markdown/PDF）
   - 实验设计
   - 结果分析
   - 结论和建议

2. **结果数据**
   - `performance_matrix.csv`
   - `statistical_tests.csv`
   - `completion_quality.csv`

3. **可视化图表**（高分辨率PNG/PDF）
   - 性能对比图
   - 误差分布图
   - 热力图
   - 特征重要性图

4. **代码和模型**
   - 训练脚本
   - 评估脚本
   - 最佳模型checkpoint
   - 完整配置文件

5. **演示Notebook**
   - 结果可视化
   - 交互式分析
   - 案例展示

## 风险和应对

### 风险1：计算资源不足

**应对**：
- 减少实验重复次数（5→3）
- 使用较小模型
- 分批次运行

### 风险2：结果不显著

**应对**：
- 增加数据量
- 调整超参数
- 尝试其他模型架构

### 风险3：两种方法性能接近

**应对**：
- 深入分析细微差异
- 考虑其他因素（效率、可解释性）
- 可能两者都可用，提供ensemble选项

## 下一步

开始实施：
```bash
# 1. 运行快速验证
python quick_experiment.py

# 2. 如果符合预期，运行完整实验
python run_full_experiment.py --config config.yaml

# 3. 分析结果
python analyze_results.py --results_dir results/
```

