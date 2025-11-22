"""
12步(120天)真双模态直接分类
使用预缓存的图像，视觉+语言+对比学习
这是创新点！
"""
import sys
import os
import argparse
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import json

from models.timesclip_classifier import TimesCLIPClassifier
from experiments.classification.data_loader_classification_cached import CachedImageClassificationDataset
from experiments.classification.improved_losses import CombinedEarlyLoss
from experiments.classification.data_augmentation import get_augmentation_pipeline
from experiments.classification.experiment_tracker import ExperimentTracker
from sklearn.model_selection import train_test_split as sk_train_test_split


def train_epoch(model, train_loader, optimizer, criterion, device, contrastive_weight=0.1,
                ts_augmentor=None, img_augmentor=None, aug_config=None):
    """训练一个epoch（带数据增强）"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, y, cached_images in tqdm(train_loader, desc='Training'):
        x, y = x.to(device), y.to(device)
        cached_images = cached_images.to(device)
        
        # ========== 数据增强 ==========
        if ts_augmentor is not None and aug_config is not None:
            # 时序数据增强（应用于语言分支）
            x = ts_augmentor(x, 
                           augment_prob=aug_config['ts_prob'],
                           augment_types=aug_config['ts_types'])
        
        if img_augmentor is not None and aug_config is not None:
            # 图像数据增强（应用于视觉分支）
            cached_images = img_augmentor(cached_images, 
                                         augment_prob=aug_config['img_prob'])
        # ==============================
        
        optimizer.zero_grad()
        
        # 前向传播 - 双模态 + 对比学习
        logits, contrastive_loss = model(x, return_contrastive_loss=True, cached_images=cached_images)
        
        # 计算损失（CombinedEarlyLoss返回tuple，但我们有现成的contrastive_loss）
        # 直接使用分类损失 + 对比学习损失
        loss_output, loss_dict = criterion(logits, y, None, None, time_ratio=1.0)
        
        # 添加对比学习损失（模型已经计算好了）
        total_loss_value = loss_output + contrastive_weight * contrastive_loss
        
        total_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_value.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc


def evaluate(model, val_loader, criterion, device, contrastive_weight=0.1):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, cached_images in val_loader:
            x, y = x.to(device), y.to(device)
            cached_images = cached_images.to(device)
            
            logits, contrastive_loss = model(x, return_contrastive_loss=True, cached_images=cached_images)
            
            # 分类损失
            loss_output, loss_dict = criterion(logits, y, None, None, time_ratio=1.0)
            
            # 总损失（包含对比学习）
            total_loss_value = loss_output + contrastive_weight * contrastive_loss
            
            total_loss += total_loss_value.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    return avg_loss, f1, acc, f1_per_class.tolist()


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description='12步双模态直接分类训练（支持断点续训）')
    parser.add_argument('--resume', action='store_true', 
                        help='从最新checkpoint恢复训练')
    args = parser.parse_args()
    
    # ============ 设置随机种子（可重复性） ============
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    import random
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # ============ 配置 ============
    TIME_STEPS = 12
    N_VARIATES = 14
    NUM_CLASSES = 4
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    PATIENCE = 15
    CONTRASTIVE_WEIGHT = 0.1
    
    # 创建独立保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果是resume，尝试找到最新的训练目录
    if args.resume:
        import glob
        pattern = f"experiments/classification/timesclip_{TIME_STEPS}steps_dual_*"
        existing_dirs = sorted(glob.glob(pattern))
        if existing_dirs:
            save_dir = existing_dirs[-1]  # 使用最新的目录
            print(f"\n✓ 找到现有训练目录: {save_dir}")
        else:
            save_dir = f"experiments/classification/timesclip_{TIME_STEPS}steps_dual_{timestamp}"
            print(f"\n⚠ 未找到现有目录，创建新目录: {save_dir}")
    else:
        save_dir = f"experiments/classification/timesclip_{TIME_STEPS}steps_dual_{timestamp}"
    
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/results", exist_ok=True)
    
    print("="*70)
    print(f"12步(120天)真双模态直接分类 [创新点]")
    print("="*70)
    print(f"保存目录: {save_dir}")
    print(f"时间步数: {TIME_STEPS} ({TIME_STEPS*10}天)")
    print(f"模态: 双模态 (视觉 + 语言)")
    print(f"对比学习: 是 (InfoNCE)")
    print(f"图像来源: 预缓存 (data/multiscale_image_cache/time_12)")
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
        "use_contrastive": True,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "patience": PATIENCE,
        "dropout": 0.3,
        "weight_decay": 5e-4,
        "focal_gamma": 2.0,
        "augmentation_mode": "medium",
        "training_time": timestamp,
        "image_cache": "data/multiscale_image_cache/time_12",
        "resume": args.resume,
        "description": "12步真双模态 - 视觉+语言+对比学习+数据增强，使用预缓存图像，创新点"
    }
    
    with open(f"{save_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # ============ 数据加载 ============
    print("\n加载数据（含预缓存图像）...")
    
    # 读取标签进行分层划分
    import pandas as pd
    df = pd.read_csv("../../data/2018four.csv")
    labels = df.iloc[:, -1].values
    indices = np.arange(len(labels))
    
    # 分层划分: 80% train+val, 20% test
    train_val_indices, test_indices = sk_train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 再从train_val中分出10%作为验证集
    train_labels = labels[train_val_indices]
    train_indices, val_indices = sk_train_test_split(
        train_val_indices, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本")
    print(f"  测试集: {len(test_indices)} 样本")
    
    # 创建数据集（手动指定time_steps和n_variates）
    CSV_PATH = "../../data/2018four.csv"
    CACHE_DIR = f"../../data/multiscale_image_cache/time_{TIME_STEPS}"  # ← 直接指向time_12目录
    
    print(f"\n创建数据集（使用 {TIME_STEPS} 步数据）...")
    train_dataset = CachedImageClassificationDataset(
        CSV_PATH, 
        indices=train_indices,
        n_variates=N_VARIATES,
        n_time_steps=TIME_STEPS,  # ← 关键：指定12步
        use_cache=True,
        disk_cache_dir=CACHE_DIR,
        load_to_memory=False
    )
    
    val_dataset = CachedImageClassificationDataset(
        CSV_PATH,
        indices=val_indices,
        n_variates=N_VARIATES,
        n_time_steps=TIME_STEPS,
        use_cache=True,
        disk_cache_dir=CACHE_DIR,
        load_to_memory=False
    )
    
    test_dataset = CachedImageClassificationDataset(
        CSV_PATH,
        indices=test_indices,
        n_variates=N_VARIATES,
        n_time_steps=TIME_STEPS,
        use_cache=True,
        disk_cache_dir=CACHE_DIR,
        load_to_memory=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n数据加载完成!")
    print(f"  每个样本: {TIME_STEPS}步 × {N_VARIATES}变量")
    print(f"  预缓存图像: {N_VARIATES}张/样本 (来自 time_{TIME_STEPS}/)")
    
    # ============ 数据增强 ============
    print("\n初始化数据增强...")
    AUGMENTATION_MODE = 'medium'  # 可选: 'light', 'medium', 'heavy'
    ts_augmentor, img_augmentor, aug_config = get_augmentation_pipeline(mode=AUGMENTATION_MODE)
    print(f"增强模式: {AUGMENTATION_MODE}")
    print(f"  时序增强概率: {aug_config['ts_prob']}")
    print(f"  时序增强类型: {aug_config['ts_types']}")
    print(f"  图像增强概率: {aug_config['img_prob']}")
    
    # ============ 模型 ============
    print("\n创建模型（真双模态 + 对比学习）...")
    model = TimesCLIPClassifier(
        time_steps=TIME_STEPS,
        n_variates=N_VARIATES,
        num_classes=NUM_CLASSES,
        use_variate_selection=True,
        use_contrastive=True,  # 对比学习
        dropout=0.3  # 从0.1提升到0.3，增强正则化
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params/1e6:.2f}M")
    print(f"可训练: {trainable_params/1e6:.2f}M")
    
    # ============ 损失函数和优化器 ============
    # 计算类别权重（使用训练集标签）
    train_labels_array = labels[train_indices]
    class_counts = np.bincount(train_labels_array.astype(int))
    class_weights = len(train_labels_array) / (NUM_CLASSES * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n类别权重: {class_weights.cpu().numpy()}")
    
    criterion = CombinedEarlyLoss(
        num_classes=NUM_CLASSES,
        focal_alpha=0.25,
        focal_gamma=2.0,
        time_weight_factor=3.0,
        contrastive_weight=CONTRASTIVE_WEIGHT  # 对比学习权重
    )
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=5e-4  # 从1e-4提升到5e-4，增强正则化
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # ============ 训练循环 ============
    # 检查是否有checkpoint可以恢复
    start_epoch = 1
    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    
    checkpoint_path = f"{save_dir}/checkpoints/latest_checkpoint.pth"
    if os.path.exists(checkpoint_path) and config.get('resume', False):
        print(f"\n✓ 发现checkpoint: {checkpoint_path}")
        print("正在恢复训练...")
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            patience_counter = checkpoint.get('patience_counter', 0)
            history = checkpoint.get('history', [])
            print(f"✓ 成功恢复！从Epoch {start_epoch}继续训练")
            print(f"  最佳Val F1: {best_val_f1:.4f}")
            print(f"  已完成: {start_epoch-1}/{EPOCHS} epochs")
        except Exception as e:
            print(f"⚠ 恢复失败: {e}")
            print("将从头开始训练...")
            start_epoch = 1
    
    print("\n开始训练（双模态 + 对比学习）...")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*70}")
        
        # 训练（带数据增强）
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, CONTRASTIVE_WEIGHT,
            ts_augmentor=ts_augmentor, img_augmentor=img_augmentor, aug_config=aug_config
        )
        
        # 验证
        val_loss, val_f1, val_acc, val_f1_per_class = evaluate(
            model, val_loader, criterion, device, CONTRASTIVE_WEIGHT
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
        
        # 每个epoch保存latest checkpoint（用于断点续训）
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'patience_counter': patience_counter,
            'history': history,
            'config': config
        }
        latest_path = f"{save_dir}/checkpoints/latest_checkpoint.pth"
        torch.save(latest_checkpoint, latest_path)
    
    # ============ 测试 ============
    print("\n" + "="*70)
    print("在测试集上评估最佳模型...")
    print("="*70)
    
    # 加载最佳模型
    checkpoint = torch.load(f"{save_dir}/checkpoints/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_acc, test_f1_per_class = evaluate(
        model, test_loader, criterion, device, CONTRASTIVE_WEIGHT
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
    
    # ============ 记录到实验追踪系统 ============
    print("\n记录实验到追踪系统...")
    tracker = ExperimentTracker()
    
    # 准备实验配置
    exp_config = {
        'description': config.get('description', f'{TIME_STEPS}步双模态分类'),
        'tags': ['dual_modal', 'contrastive_learning', f'aug_{AUGMENTATION_MODE}'],
        'time_steps': TIME_STEPS,
        'dropout': 0.3,
        'weight_decay': 5e-4,
        'use_contrastive': True,
        'contrastive_weight': CONTRASTIVE_WEIGHT,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'epochs': EPOCHS,
        'augmentation_mode': AUGMENTATION_MODE,
        'ts_aug_prob': aug_config['ts_prob'],
        'img_aug_prob': aug_config['img_prob'],
        'aug_types': ','.join(aug_config['ts_types']),
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'model_type': 'dual'
    }
    
    # 准备实验结果
    exp_results = {
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'best_epoch': best_epoch,
        'best_train_f1': history[best_epoch-1]['train_f1'],
        'best_train_acc': history[best_epoch-1]['train_acc'],
        'best_val_f1': best_val_f1,
        'best_val_acc': history[best_epoch-1]['val_acc'],
        'final_test_f1': test_f1,
        'final_test_acc': test_acc,
        'class_f1': test_f1_per_class,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time_hours': (datetime.now() - datetime.strptime(timestamp, "%Y%m%d_%H%M%S")).total_seconds() / 3600
    }
    
    # 记录实验
    exp_id = tracker.log_experiment(
        exp_config, 
        exp_results, 
        notes=f"数据增强模式: {AUGMENTATION_MODE}, Dropout: 0.3, WD: 5e-4"
    )
    
    print(f"\n{'='*70}")
    print(f"训练完成! [双模态创新点]")
    print(f"实验ID: {exp_id}")
    print(f"模型保存在: {save_dir}")
    print(f"特点: 视觉+语言+对比学习+数据增强")
    print(f"Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}")
    print(f"过拟合差距: {exp_results['best_train_f1'] - best_val_f1:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

