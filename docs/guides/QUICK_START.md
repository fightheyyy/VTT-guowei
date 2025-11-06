# TimesCLIP 快速开始指南

## 🎯 项目目标

**输入**: 2025年1-5月的多波段遥感数据  
**输出**: 2025年产量预测值

---

## 📋 准备工作

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n vtt python=3.10 -y
conda activate vtt

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保有以下文件：
- ✅ `extract2022_20251010_165007.csv` - 2022年训练数据（已有）

---

## 🚀 三步走

### 步骤1: 测试数据加载

```bash
python test_data_loading.py
```

**预期输出**:
```
✓ 数据加载成功！
  - 波段数: 7
  - 训练批次数: 25
  - 测试批次数: 7
```

### 步骤2: 训练两阶段模型

```bash
python train_two_stage.py
```

**这会做什么？**
- 阶段1: 训练TimesCLIP（波段值补全）- 约30-60分钟
- 阶段2: 训练YieldPredictor（产量预测）- 约20-40分钟

**预期输出**:
```
阶段1完成！最佳验证损失: XXXX
阶段2完成！最佳验证损失: XXXX
```

### 步骤3: 预测2025年产量

```bash
python predict_2025.py
```

**预期输出**:
```
2025年预测产量: XX.XX
✓ 可视化结果已保存: 2025_prediction.png
```

---

## 📊 监控训练

在训练过程中，可以实时查看训练进度：

```bash
# 新开一个终端
tensorboard --logdir=logs
```

然后在浏览器打开 `http://localhost:6006`

---

## 🔧 自定义配置

### 方法1: 灵活输入月份 ⭐ 新功能！

**可以用更少的月份进行预测！**（精度会有影响）

```bash
# 用1-2月预测全年（最早，精度较低）
python train_flexible.py --input_months 2

# 用1-3月预测全年（较早，精度中等）
python train_flexible.py --input_months 3

# 用1-5月预测全年（推荐，平衡时效和精度）
python train_flexible.py --input_months 5

# 用1-6月预测剩余月份（较晚，精度高）
python train_flexible.py --input_months 6
```

然后用对应的月份数预测：
```bash
python predict_flexible.py --input_months 3 --visualize
```

📖 详细说明见：`FLEXIBLE_CONFIG_GUIDE.md`

### 方法2: 修改训练参数

编辑 `train_two_stage.py`:

```python
# 阶段1配置
train_stage1_timeseries(
    lookback=18,           # 输入长度（1-5月）
    prediction_steps=18,   # 预测长度（6-12月）
    batch_size=16,         # 批次大小
    epochs=50,             # 训练轮数
    d_model=256,           # 模型大小（256快/512准）
)

# 阶段2配置
train_stage2_yield(
    batch_size=32,
    epochs=100,
    d_model=256,
)
```

### 选择波段

```python
# 使用7个常用波段（默认）
selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']

# 或使用全部14个波段
selected_bands = None
```

---

## 📁 项目文件结构

```
VTT/
├── models/                          # 模型定义
│   ├── timesclip.py                # 阶段1: 时间序列补全
│   ├── yield_predictor.py          # 阶段2: 产量预测
│   └── ...其他组件
│
├── checkpoints/                     # 训练好的模型
│   ├── stage1_timeseries_best.pth  # 阶段1最佳模型
│   └── stage2_yield_best.pth       # 阶段2最佳模型
│
├── logs/                            # TensorBoard日志
│   ├── stage1/
│   └── stage2/
│
├── train_two_stage.py               # 两阶段训练脚本
├── predict_2025.py                  # 2025年预测脚本
├── data_loader_with_yield.py        # 数据加载器
├── test_data_loading.py             # 数据测试
│
└── extract2022_20251010_165007.csv  # 训练数据
```

---

## 🎓 模型说明

### 阶段1: TimesCLIP（波段值补全）

**作用**: 1-5月波段值 → 6-12月波段值

```
输入: [18, 7]  (18个时间步 × 7个波段)
   ↓
TimesCLIP模型（双模态学习）
   ↓
输出: [18, 7]  (6-12月预测值)
```

### 阶段2: YieldPredictor（产量预测）

**作用**: 全年波段值 → 产量

```
输入: [36, 7]  (完整全年)
   ↓
YieldPredictor模型
   ↓
输出: 产量值（标量）
```

---

## 💡 使用真实2025年数据

当获得2025年1-5月的真实观测数据时：

```python
import pandas as pd
import numpy as np
from predict_2025 import predict_2025_yield

# 1. 读取2025年数据
df_2025 = pd.read_csv('2025_observation.csv')

# 2. 提取波段数据
band_names = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
input_2025 = []

for band in band_names:
    # 提取前18个时间步
    band_cols = [f'{band}_{i:02d}' for i in range(18)]
    band_values = df_2025.loc[0, band_cols].values
    input_2025.append(band_values)

input_2025 = np.array(input_2025).T  # [18, 7]

# 3. 预测
result = predict_2025_yield(
    input_2025_data=input_2025,
    band_names=band_names,
    device='cuda',
    visualize=True
)

print(f"2025年产量预测: {result['yield_prediction']:.2f}")
```

---

## ❓ 常见问题

### Q: 训练需要多长时间？

**A**: 
- GPU (RTX 3060): 阶段1约30分钟，阶段2约20分钟
- GPU (RTX 4090): 阶段1约15分钟，阶段2约10分钟
- CPU: 不推荐（太慢）

### Q: 显存不够怎么办？

**A**: 减小`batch_size`和`d_model`：
```python
batch_size=8      # 从16减到8
d_model=128       # 从256减到128
```

### Q: 如何提高预测精度？

**A**: 
1. 使用更多波段（全部14个）
2. 增加训练数据（多年份）
3. 增大模型（d_model=512）
4. 训练更多轮数

### Q: 可以预测其他作物吗？

**A**: 可以！只需用对应作物的训练数据重新训练模型。

---

## 📈 预期结果

### 训练完成后

**阶段1（时间序列补全）**:
- MSE损失应逐渐下降
- 预测曲线应能跟随季节性趋势
- 对比学习损失收敛到2.0以下

**阶段2（产量预测）**:
- MSE损失和MAE应稳定下降
- 在测试集上有良好的预测精度
- 预测值与真实值应有较强相关性

### 预测可视化

运行预测后会生成 `2025_prediction.png`，包含：
- 蓝色实线：已知数据（1-5月）
- 红色虚线：预测数据（6-12月）
- 每个波段一个子图

---

## 🎉 完成！

现在你已经：
- ✅ 训练了两个模型
- ✅ 可以预测2025年产量
- ✅ 理解了整个工作流程

需要帮助？查看：
- 📖 `TWO_STAGE_GUIDE.md` - 详细的两阶段指南
- 📖 `TRAINING_GUIDE.md` - 训练详细说明
- 📖 `PROJECT_SUMMARY.md` - 项目完整总结

---

## 🔄 完整命令速查

```bash
# 1. 环境准备
conda create -n vtt python=3.10 -y
conda activate vtt
pip install -r requirements.txt

# 2. 测试数据
python test_data_loading.py

# 3. 训练模型（约1-2小时）
python train_two_stage.py

# 4. 监控训练（可选，新终端）
tensorboard --logdir=logs

# 5. 预测2025年
python predict_2025.py

# 完成！查看结果：
# - 终端输出的产量值
# - 2025_prediction.png 可视化图
```

**祝预测成功！** 🌾

