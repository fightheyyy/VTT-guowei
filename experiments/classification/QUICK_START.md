# 快速开始 - 超越CLEC的早期分类

## 🎯 目标

找到最早可识别时间（F1≥0.8），超越CLEC框架的识别速度。

---

## 📋 完整流程

### 步骤1: 数据已准备好 ✅

你的图像缓存已经生成完毕，可以直接使用：
- 位置: `data/image_cache/`
- 格式: PNG图像（37个时间步完整数据）
- **无需重新准备！**

---

### 步骤2: 训练改进模型 ⭐

#### 方式A: 使用批处理文件（推荐）

```bash
cd experiments/classification
run_improved_training.bat
```

#### 方式B: 手动运行

```bash
cd experiments/classification

python train_classification_improved.py \
    --model_type language_only \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0001 \
    --time_weight_factor 2.0
```

**训练时间**: 约2-4小时（GPU）

**关键改进**:
- ✅ 时间感知Focal Loss（早期权重更高）
- ✅ 时间masking增强（动态截断序列）
- ✅ 课程学习（从长到短渐进）

---

### 步骤3: 测试早期识别时间

训练完成后，运行测试脚本：

```bash
python test_early_recognition.py \
    --model_path timesclip_improved/checkpoints/language_only_best.pth \
    --f1_threshold 0.8
```

**输出**:
```
时间步   天数     F1 (macro)   类别0    类别1    类别2    类别3    状态
------------------------------------------------------------------------
3        30       0.6234       0.59     0.61     0.64     0.66     ✗ 未达标
6        60       0.7456       0.71     0.75     0.76     0.78     ✗ 未达标
9        90       0.8123       0.79     0.81     0.82     0.83     ✓ 可识别
12       120      0.8567       0.84     0.86     0.86     0.87     ✓ 可识别
...

🎯 最早可识别时间: 9步 (90天)
   F1分数: 0.8123
```

---

## 📊 查看结果

### TensorBoard可视化

```bash
tensorboard --logdir=experiments/classification/timesclip_improved/logs
```

浏览器打开: http://localhost:6006

查看:
- 训练/验证F1曲线
- 课程学习进度
- 损失变化

### 结果文件

```
timesclip_improved/
├── results/
│   ├── language_only_results.json         # 详细结果
│   ├── language_only_confusion_matrix.png # 混淆矩阵
│   └── early_recognition_curve.png        # 早期识别曲线
└── checkpoints/
    └── language_only_best.pth             # 最佳模型
```

---

## 🔬 对比实验（可选）

### 对比标准方法

如果想对比改进效果，可以运行标准训练：

```bash
# 1. 标准方法
python train_classification_timesclip.py --model_type language_only

# 2. 改进方法  
python train_classification_improved.py --model_type language_only
```

然后对比两者的早期识别时间。

---

## 📈 预期结果

### vs CLEC对比

| 作物 | CLEC | 我们的目标 | 提升 |
|-----|------|----------|-----|
| 水稻 | 120天 | 90-100天 | ⬇️ 20-30天 |
| 大豆 | 190天 | 150-170天 | ⬇️ 20-40天 |
| 玉米 | 200天 | 160-180天 | ⬇️ 20-40天 |

### 性能指标

| 时间 | 标准方法 | 改进方法 | 提升 |
|-----|---------|---------|-----|
| 60天 | 0.68 | 0.78 | +0.10 ✅ |
| 90天 | 0.75 | 0.85 | +0.10 ✅ |
| 120天 | 0.82 | 0.90 | +0.08 ✅ |
| 完整 | 0.89 | 0.93 | +0.04 ✅ |

---

## ⚙️ 超参数调优

如果结果不理想，可以调整：

### 提升早期性能

```bash
python train_classification_improved.py \
    --time_weight_factor 3.0 \        # 增大早期权重
    --min_ratio_end 0.1               # 使用更短的序列
```

### 稳定训练

```bash
python train_classification_improved.py \
    --warmup_ratio 0.3 \              # 延长warmup
    --min_ratio_start 0.8             # 从更长序列开始
```

### 快速测试

```bash
python train_classification_improved.py \
    --epochs 30 \                     # 减少轮数
    --batch_size 128                  # 增大batch
```

---

## 🐛 常见问题

### Q1: 训练很慢

**A**: 
1. 确认在用GPU: `--device cuda`
2. 增大batch_size: `--batch_size 128`
3. 减少验证频率

### Q2: 内存不足

**A**:
1. 减小batch_size: `--batch_size 32`
2. 确认使用动态读取: `load_to_memory=False`

### Q3: F1不收敛

**A**:
1. 增加warmup: `--warmup_ratio 0.3`
2. 降低学习率: `--lr 0.00005`
3. 检查数据是否正确加载

### Q4: 早期F1太低

**A**:
1. 增大时间权重: `--time_weight_factor 3.0`
2. 使用更激进的课程: `--min_ratio_end 0.15`
3. 训练更多轮数: `--epochs 150`

---

## 📝 实验记录模板

训练完成后记录：

```markdown
## 实验记录

### 基本信息
- 日期: 2024-XX-XX
- 模型: TimesCLIP (language_only)
- 数据: 2018four.csv (5557样本)

### 超参数
- epochs: 100
- batch_size: 64
- lr: 1e-4
- time_weight_factor: 2.0
- min_ratio: 0.7 -> 0.2

### 结果
- 最早识别时间: XX天
- F1分数: 0.XX
- 测试准确率: 0.XX

### 对比CLEC
- 提前天数: XX天
- F1提升: +0.XX

### 结论
（你的发现和分析）
```

---

## 🚀 下一步

完成基础实验后，可以尝试：

1. **双模态**: 实现视觉+语言的对比学习
2. **集成学习**: 多个时间长度模型集成
3. **迁移学习**: 在其他数据集上测试
4. **理论分析**: 为什么改进有效

---

## 💬 需要帮助？

查看详细文档：
- 策略说明: `STRATEGY_IMPROVEMENTS.md`
- 实施指南: `IMPLEMENTATION_GUIDE.md`
- 损失函数: `improved_losses.py`

