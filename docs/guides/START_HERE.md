# 🎯 从这里开始！

## 最快开始方式

### 方法1：一键运行（Windows）⭐

**双击运行**：
```
run_all.bat
```

这会自动执行：
1. ✅ 测试数据
2. ✅ 训练模型
3. ✅ 测试预测

全自动，只需等待完成！

---

### 方法2：手动运行（更灵活）

#### 第1步：测试数据
```bash
python test_data_loading.py
```

#### 第2步：训练
```bash
python train_two_stage.py
```

#### 第3步：预测
```bash
python predict_2025.py
```

---

## ⚡ 快速命令

```bash
# 最基础的三行命令
python test_data_loading.py       # 1. 测试
python train_two_stage.py         # 2. 训练（1-2小时）
python predict_2025.py             # 3. 预测

# 完成！
```

---

## 🎓 如果想了解更多

- **快速开始**: `QUICK_START.md`
- **灵活配置**: `FLEXIBLE_CONFIG_GUIDE.md`
- **输入月份对比**: `INPUT_MONTHS_COMPARISON.md`
- **两阶段详解**: `TWO_STAGE_GUIDE.md`

---

## ⏱️ 预计时间

| 步骤 | 时间 |
|------|------|
| 测试数据 | <1分钟 |
| 训练模型 | 1-2小时 |
| 测试预测 | <1分钟 |
| **总计** | **1-2小时** |

---

## ✅ 完成后

你会得到：
- ✅ 两个训练好的模型
- ✅ 可以预测2025年产量的能力
- ✅ 训练过程的可视化日志

---

## 🆘 遇到问题？

1. **CUDA不可用**：会自动切换到CPU（会很慢）
2. **内存不足**：减小batch_size
3. **其他错误**：查看错误信息，或查阅文档

**现在就开始吧！运行 `run_all.bat` 或执行三行命令！**

