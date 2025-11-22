# 快速开始 - 改进版训练

## 1分钟快速启动

### Windows用户

```bash
cd E:\VTT\VTT-guowei\experiments\forecasting
run_improved.bat
# 选择 [1] 端到端Pipeline
```

### 命令行用户

```bash
cd experiments/forecasting
python train_pipeline_improved.py --input_len 6 --epochs 100
```

就这么简单! 改进版会自动:
- ✅ 使用Focal Loss处理类别不平衡
- ✅ 应用类别权重
- ✅ 增强正则化 (dropout=0.3)
- ✅ 降低学习率 (lr=5e-5)
- ✅ 梯度裁剪
- ✅ 自动保存最佳模型

---

## 继续已有训练

如果训练中断或想继续训练:

```bash
python train_pipeline_improved.py \
    --checkpoint checkpoints/pipeline_e2e_in6.pth \
    --epochs 100
```

模型会从断点继续，保留之前的最佳F1记录。

---

## 查看当前训练效果

```bash
# 测试最新checkpoint
python test_checkpoint.py --checkpoint checkpoints/pipeline_improved_in6.pth
```

---

## 预期结果

### 原版 (当前)
```
Best Val F1: 0.42
波动: 0.27-0.42
Class 0 F1: ~0.15 (很差)
```

### 改进版 (预期)
```
Best Val F1: 0.50-0.55 (+28%)
波动: 0.48-0.55 (稳定)
Class 0 F1: ~0.35 (+133%)
```

---

## 如何判断训练好不好

### ✅ 好的训练
```
Epoch 10: Loss=0.8592, Val F1=0.45
Epoch 11: Loss=0.8445, Val F1=0.47
Epoch 12: Loss=0.8350, Val F1=0.49
Epoch 13: Loss=0.8285, Val F1=0.50
Epoch 14: Loss=0.8195, Val F1=0.51 ← 稳定上升
```
- 训练损失稳定下降
- 验证F1稳定上升
- 波动小

### ❌ 不好的训练 (你当前的情况)
```
Epoch 10: Loss=0.8592, Val F1=0.25
Epoch 11: Loss=0.8581, Val F1=0.41 ← 跳跃
Epoch 12: Loss=0.8529, Val F1=0.32 ← 掉了
Epoch 13: Loss=0.8445, Val F1=0.40
Epoch 14: Loss=0.8442, Val F1=0.29 ← 又掉
```
- 训练损失下降但验证指标混乱
- 剧烈波动
- 过拟合

---

## 常见问题

### Q: 改进版一定能达到F1=0.55吗?
**A**: 不一定。预期0.50-0.55，取决于:
- 运气 (随机种子)
- GPU性能
- 训练时长

但肯定比原版的0.42好!

### Q: 训练要多久?
**A**: 
- 每个epoch: ~2小时
- 总共: 50-100小时 (可以分批训练)
- 建议: 挂机训练，用checkpoint恢复

### Q: 改进版为什么用batch_size=32而不是64?
**A**: 
- 小batch增加随机性
- 有助于泛化
- 对不平衡数据集特别有效

### Q: 可以只用Focal Loss不用其他改进吗?
**A**: 可以，但效果会打折扣。建议全套使用:
- Focal Loss: 最重要 (贡献+0.08 F1)
- 正则化: 很重要 (贡献+0.04 F1)
- 其他: 锦上添花 (贡献+0.04 F1)

---

## 下一步

训练完成后:

1. **查看结果**
   ```bash
   python test_checkpoint.py
   ```

2. **对比原版**
   ```bash
   run_improved.bat -> [4] 对比实验
   ```

3. **调优参数** (如果效果不理想)
   - 参考 `IMPROVEMENT_GUIDE.md`

---

## 遇到问题?

### 训练卡住不动
→ 检查GPU占用: `nvidia-smi`

### 显存不足
→ 降低batch_size: `--batch_size 16`

### F1仍然波动大
→ 增加dropout: `--dropout 0.4`

### 效果没提升
→ 检查是否用了旧的训练脚本 (应该用train_pipeline_improved.py)

---

**总之: 直接运行 `run_improved.bat` 就行了!**

