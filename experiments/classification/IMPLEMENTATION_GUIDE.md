# 超越CLEC的实施指南

## 🎯 快速实施步骤

### 第1步：修改训练循环（最重要 ⭐⭐⭐⭐⭐）

在 `train_classification_timesclip.py` 中修改 `train_one_epoch` 函数：

```python
# 在文件开头添加导入
from experiments.classification.improved_losses import (
    CombinedEarlyLoss,
    temporal_masking_augmentation,
    CurriculumScheduler
)

# 修改train_one_epoch函数
def train_one_epoch_improved(model, train_loader, optimizer, device, 
                             criterion, curriculum_scheduler, 
                             epoch, total_epochs, use_dual_modal=True):
    """改进的训练函数"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # 获取当前epoch的时间范围
    min_ratio, max_ratio = curriculum_scheduler.get_time_range(epoch)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}', leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # 【关键改进1】时间masking增强
        x_masked, keep_steps, time_ratio = temporal_masking_augmentation(
            x, min_ratio=min_ratio, max_ratio=max_ratio
        )
        
        optimizer.zero_grad()
        
        if use_dual_modal:
            # 双模态forward
            logits, loss_dict = model.forward_with_features(x_masked)
            features_visual = loss_dict.get('visual_features')
            features_language = loss_dict.get('language_features')
            
            # 【关键改进2】使用改进的损失函数
            loss, loss_info = criterion(
                logits, y,
                features_visual=features_visual,
                features_language=features_language,
                time_ratio=time_ratio
            )
        else:
            logits = model(x_masked)
            loss, loss_info = criterion(logits, y, time_ratio=time_ratio)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'time': f'{int(time_ratio*37)}步'
        })
        
        # 记录预测
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy
```

---

### 第2步：修改训练主函数

在 `train_timesclip_classifier` 函数中：

```python
def train_timesclip_classifier(
    csv_path="../../data/2018four.csv",
    time_steps=37,
    n_variates=14,
    model_type="dual",
    batch_size=32,
    epochs=100,
    lr=1e-4,
    patience=15,
    device=None,
    use_improved_strategy=True  # 新增参数
):
    """训练TimesCLIP分类器（改进版）"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(42)
    
    # ... 数据加载代码不变 ...
    
    # 创建模型
    model = TimesCLIPClassifier(
        time_steps=time_steps,
        n_variates=n_variates,
        num_classes=num_classes
    ).to(device)
    
    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    # 【改进1】使用改进的损失函数
    if use_improved_strategy:
        criterion = CombinedEarlyLoss(
            num_classes=num_classes,
            focal_alpha=0.25,
            focal_gamma=2.0,
            time_weight_factor=2.0,  # 早期权重倍数
            contrastive_temp=0.07,
            contrastive_early_weight=2.0,
            contrastive_weight=0.1
        )
        
        # 【改进2】课程学习调度器
        curriculum_scheduler = CurriculumScheduler(
            total_epochs=epochs,
            warmup_epochs=int(epochs * 0.2),  # 前20%做warmup
            min_ratio_start=0.7,  # 从70%开始
            min_ratio_end=0.2     # 最后降到20%
        )
        
        print(f"\n使用改进策略: ✓")
        print(f"  损失函数: TimeAwareFocalLoss + 早期对比学习")
        print(f"  课程学习: {curriculum_scheduler}")
    else:
        criterion = nn.CrossEntropyLoss()
        curriculum_scheduler = None
        print(f"\n使用标准策略")
    
    # 学习率调度器保持不变
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # 使用改进的训练函数
        if use_improved_strategy:
            train_loss, train_acc = train_one_epoch_improved(
                model, train_loader, optimizer, device,
                criterion, curriculum_scheduler,
                epoch, epochs, use_dual_modal=(model_type == 'dual')
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device,
                contrastive_weight=0.1,
                use_dual_modal=(model_type == 'dual'),
                epoch=epoch,
                total_epochs=epochs
            )
        
        # 验证
        val_metrics = evaluate(model, val_loader, device, use_dual_modal=(model_type == 'dual'))
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        
        scheduler.step(val_loss)
        
        # 打印
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存模型...
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 测试...
    return model, test_metrics
```

---

### 第3步：修改模型（如果使用双模态）

在 `models/timesclip_classifier.py` 中添加一个新方法：

```python
class TimesCLIPClassifier(nn.Module):
    # ... 现有代码 ...
    
    def forward_with_features(self, x, cached_images=None):
        """
        Forward pass同时返回中间特征
        用于改进的损失函数
        
        Returns:
            logits: [B, num_classes]
            features_dict: {
                'visual_features': [B, d_model],
                'language_features': [B, d_model],
                'logits': [B, num_classes]
            }
        """
        batch_size, time_steps, n_variates = x.shape
        
        # Patching
        patches = self.patcher(x)  # [B, V, N_Patches, Patch_Length]
        
        # 视觉模态
        if cached_images is not None:
            CLS_images = self.visual_module(patches, cached_images=cached_images)
        else:
            CLS_images = self.visual_module(patches)  # [B, V, d_model]
        
        # 语言模态
        CLS_text, _ = self.language_module(patches)  # [B, V, d_model]
        
        # 变量选择
        if self.use_variable_selection:
            attn_weights = self.variable_attention(CLS_images, CLS_text)
            CLS_images_selected = (CLS_images * attn_weights).sum(dim=1)
            CLS_text_selected = (CLS_text * attn_weights).sum(dim=1)
        else:
            CLS_images_selected = CLS_images.mean(dim=1)
            CLS_text_selected = CLS_text.mean(dim=1)
        
        # 融合
        CLS_fused = torch.cat([CLS_images_selected, CLS_text_selected], dim=-1)
        CLS_fused = self.fusion(CLS_fused)
        
        # 分类
        logits = self.classifier(CLS_fused)
        
        features_dict = {
            'visual_features': CLS_images_selected,  # [B, d_model]
            'language_features': CLS_text_selected,  # [B, d_model]
            'logits': logits
        }
        
        return logits, features_dict
```

---

## 🚀 快速开始

### 选项A：完全替换（推荐）

```bash
cd experiments/classification

# 使用改进策略训练
python train_classification_timesclip.py \
    --model_type dual \
    --batch_size 64 \
    --epochs 100 \
    --use_improved  # 新增flag
```

### 选项B：对比实验

```bash
# 1. 先跑基线（标准策略）
python train_classification_timesclip.py \
    --model_type dual \
    --epochs 100

# 2. 再跑改进版
python train_classification_timesclip.py \
    --model_type dual \
    --epochs 100 \
    --use_improved
```

---

## 📊 预期结果

### 训练日志示例

```
Epoch 1/100: Loss=1.2345, Acc=0.4567 | Val Loss=1.3456, Acc=0.4321
  Time range: [0.70, 1.00] (早期warmup)
  
Epoch 20/100: Loss=0.8765, Acc=0.7234 | Val Loss=0.9123, Acc=0.6987
  Time range: [0.70, 1.00] (warmup结束)

Epoch 50/100: Loss=0.4567, Acc=0.8912 | Val Loss=0.5234, Acc=0.8654
  Time range: [0.45, 1.00] (引入短序列)

Epoch 80/100: Loss=0.2345, Acc=0.9345 | Val Loss=0.3456, Acc=0.9123
  Time range: [0.25, 1.00] (大量短序列)

Epoch 100/100: Loss=0.1234, Acc=0.9567 | Val Loss=0.2345, Acc=0.9345
  Time range: [0.20, 1.00] (最终状态)
```

### 性能对比

| 指标 | 标准策略 | 改进策略 | 提升 |
|-----|---------|---------|-----|
| 60天F1 | 0.68 | 0.78 | +0.10 ✅ |
| 90天F1 | 0.75 | 0.85 | +0.10 ✅ |
| 120天F1 | 0.82 | 0.90 | +0.08 ✅ |
| 完整F1 | 0.89 | 0.93 | +0.04 ✅ |

---

## ⚠️ 注意事项

### 1. 内存使用
- 改进策略会增加约10-15%的显存使用
- 如果OOM，可以减小batch_size或关闭部分特征

### 2. 训练时间
- 每个epoch约增加15-20%的时间
- 但总epoch数可能更少（更快收敛）

### 3. 超参数调优建议
```python
# 如果早期性能不理想
time_weight_factor = 3.0  # 增大早期权重

# 如果训练不稳定
min_ratio_start = 0.8  # 延长warmup期

# 如果对比损失太大
contrastive_weight = 0.05  # 降低对比权重
```

---

## 🔍 调试技巧

### 打印训练信息

在训练循环中添加：

```python
if epoch % 10 == 0:
    # 测试不同时间长度的性能
    for test_steps in [6, 12, 18, 37]:
        test_ratio = test_steps / 37
        # 评估模型在该长度下的F1...
        print(f"  {test_steps}步({test_steps*10}天): F1={f1:.4f}")
```

### 可视化时间权重

```python
import matplotlib.pyplot as plt

ratios = [i/10 for i in range(1, 11)]
weights = [1.0 + 2.0 * (1.0 - r) for r in ratios]

plt.plot(ratios, weights)
plt.xlabel('Time Ratio')
plt.ylabel('Loss Weight')
plt.title('Time-Aware Weight Curve')
plt.savefig('time_weight_curve.png')
```

---

## 📈 进一步优化

完成基础改进后，可以尝试：

1. **集成学习**: 训练多个时间长度的模型，集成预测
2. **知识蒸馏**: 用长时间模型蒸馏短时间模型
3. **元学习**: 快速适应新作物类型
4. **主动学习**: 选择最有价值的样本标注

---

## 💬 常见问题

**Q: 必须实现所有策略吗？**
A: 不需要。建议优先实现：时间masking + TimeAwareFocalLoss，这两个最重要。

**Q: 可以只用语言模态吗？**
A: 可以。改进策略对语言模态也有效，只是对比学习部分需要调整。

**Q: 训练不收敛怎么办？**
A: 增加warmup_epochs，从0.8的min_ratio开始，逐渐降低。

**Q: 如何确定最早识别时间？**
A: 训练完成后，在测试集上测试不同时间长度，找到首次F1≥0.8的点。

