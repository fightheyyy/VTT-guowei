"""
12步(120天)独立训练脚本
完全独立，不覆盖任何现有模型
"""
import sys
import os
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import json

from models.timesclip_classifier import TimesCLIPClassifier
from experiments.classification.data_loader_classification import create_classification_dataloaders
from experiments.classification.improved_losses import CombinedEarlyLoss


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, y in tqdm(train_loader, desc='Training'):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(x, return_contrastive_loss=False)
        
        # 计算损失
        loss = criterion(logits, y, None, None)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc


def evaluate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x, return_contrastive_loss=False)
            loss = criterion(logits, y, None, None)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    # 各类别F1
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    return avg_loss, f1, acc, f1_per_class.tolist()


def main():
    # ============ 配置 ============
    TIME_STEPS = 12
    N_VARIATES = 14
    NUM_CLASSES = 4
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    PATIENCE = 15
    
    # 创建独立保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/classification/timesclip_{TIME_STEPS}steps_{timestamp}"
    
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/results", exist_ok=True)
    
    print("="*70)
    print(f"12步(120天)直接分类训练 - 独立版本")
    print("="*70)
    print(f"保存目录: {save_dir}")
    print(f"时间步数: {TIME_STEPS} ({TIME_STEPS*10}天)")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print("="*70)
    
    # 保存配置
    config = {
        "time_steps": TIME_STEPS,
        "days": TIME_STEPS * 10,
        "n_variates": N_VARIATES,
        "num_classes": NUM_CLASSES,
        "model_type": "dual",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "patience": PATIENCE,
        "focal_gamma": 2.0,
        "training_time": timestamp,
        "description": "12步直接分类，独立保存，不覆盖其他模型"
    }
    
    with open(f"{save_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # ============ 数据加载 ============
    print("\n加载数据...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        create_classification_dataloaders(
            csv_path="../../data/2018four.csv",
            batch_size=BATCH_SIZE,
            n_variates=N_VARIATES,
            n_time_steps=TIME_STEPS,  # ← 关键：使用12步
            stratify=True,
            shuffle=True
        )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # ============ 模型 ============
    print("\n创建模型...")
    model = TimesCLIPClassifier(
        time_steps=TIME_STEPS,
        n_variates=N_VARIATES,
        num_classes=NUM_CLASSES,
        use_variate_selection=True,
        use_contrastive=True,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params/1e6:.2f}M")
    print(f"可训练: {trainable_params/1e6:.2f}M")
    
    # ============ 损失函数和优化器 ============
    # 计算类别权重
    train_labels = np.array([label for _, label in train_dataset])
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (NUM_CLASSES * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n类别权重: {class_weights.cpu().numpy()}")
    
    criterion = CombinedEarlyLoss(
        num_classes=NUM_CLASSES,
        focal_alpha=0.25,
        focal_gamma=2.0,
        time_weight_factor=3.0,
        contrastive_weight=0.1,
        class_weights=class_weights
    )
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # ============ 训练循环 ============
    print("\n开始训练...")
    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*70}")
        
        # 训练
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # 验证
        val_loss, val_f1, val_acc, val_f1_per_class = evaluate(
            model, val_loader, criterion, device
        )
        
        # 学习率调整
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        epoch_result = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_f1_per_class': val_f1_per_class,
            'lr': current_lr
        }
        history.append(epoch_result)
        
        print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }
            
            save_path = f"{save_dir}/checkpoints/best_model.pth"
            torch.save(checkpoint, save_path)
            print(f"  [√] 保存最佳模型 (Val F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n早停: {PATIENCE} epochs没有改善")
                break
    
    # ============ 测试 ============
    print("\n" + "="*70)
    print("在测试集上评估最佳模型...")
    print("="*70)
    
    # 加载最佳模型
    checkpoint = torch.load(f"{save_dir}/checkpoints/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_acc, test_f1_per_class = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n测试集结果:")
    print(f"  F1 Score (macro): {test_f1:.4f}")
    print(f"  准确率: {test_acc:.4f}")
    print(f"\n各类别F1:")
    for i, f1 in enumerate(test_f1_per_class):
        print(f"    Class {i}: {f1:.4f}")
    
    # 保存结果
    results = {
        'config': config,
        'best_epoch': checkpoint['epoch'],
        'best_val_f1': best_val_f1,
        'test_metrics': {
            'f1_macro': test_f1,
            'accuracy': test_acc,
            'f1_per_class': test_f1_per_class
        },
        'training_history': history
    }
    
    with open(f"{save_dir}/results/test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"训练完成!")
    print(f"模型保存在: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

