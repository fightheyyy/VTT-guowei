"""
安全对比不同时间步
每个模型保存到独立目录，带时间戳，互不覆盖
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
from sklearn.metrics import f1_score, accuracy_score
import json
import matplotlib.pyplot as plt

from models.timesclip_classifier import TimesCLIPClassifier
from experiments.classification.data_loader_classification import create_classification_dataloaders
from experiments.classification.improved_losses import CombinedEarlyLoss

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def train_single_timestep(time_steps, save_base_dir, epochs=50):
    """训练单个时间步模型"""
    print(f"\n{'='*70}")
    print(f"训练 {time_steps} 步 ({time_steps*10} 天) 模型")
    print(f"{'='*70}\n")
    
    # 创建独立保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_base_dir}/timestep_{time_steps}steps_{timestamp}"
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/results", exist_ok=True)
    
    print(f"保存到: {save_dir}\n")
    
    # 配置
    config = {
        "time_steps": time_steps,
        "days": time_steps * 10,
        "batch_size": 64,
        "epochs": epochs,
        "lr": 1e-4,
        "training_time": timestamp
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        create_classification_dataloaders(
            csv_path="../../data/2018four.csv",
            batch_size=64,
            n_variates=14,
            n_time_steps=time_steps,
            stratify=True,
            shuffle=True
        )
    
    # 模型
    model = TimesCLIPClassifier(
        time_steps=time_steps,
        n_variates=14,
        num_classes=4,
        use_variate_selection=True,
        use_contrastive=True,
        dropout=0.1
    ).to(device)
    
    # 损失和优化器
    train_labels = np.array([label for _, label in train_dataset])
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (4 * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = CombinedEarlyLoss(
        num_classes=4,
        focal_alpha=0.25,
        focal_gamma=2.0,
        time_weight_factor=3.0,
        contrastive_weight=0.1,
        class_weights=class_weights
    )
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 训练
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, return_contrastive_loss=False)
            loss = criterion(logits, y, None, None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, return_contrastive_loss=False)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)
        
        scheduler.step(val_f1)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
        
        # 保存最佳
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, f"{save_dir}/checkpoints/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停于Epoch {epoch}")
                break
    
    # 测试
    checkpoint = torch.load(f"{save_dir}/checkpoints/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, return_contrastive_loss=False)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())
    
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1_per_class = f1_score(test_labels, test_preds, average=None).tolist()
    
    results = {
        'config': config,
        'save_dir': save_dir,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'test_f1_per_class': test_f1_per_class
    }
    
    with open(f"{save_dir}/results/test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  测试集: F1={test_f1:.4f}, Acc={test_acc:.4f}")
    print(f"  已保存到: {save_dir}")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("安全对比不同时间步 - 每个模型独立保存")
    print("="*70)
    
    # 创建总对比目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = f"experiments/classification/timesteps_comparison_{timestamp}"
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"\n所有模型保存在: {comparison_dir}")
    print("每个时间步有独立子目录，互不覆盖\n")
    
    # 测试的时间步
    test_timesteps = [6, 12, 18, 37]
    all_results = {}
    
    for ts in test_timesteps:
        try:
            results = train_single_timestep(ts, comparison_dir, epochs=50)
            all_results[ts] = results
        except Exception as e:
            print(f"训练{ts}步时出错: {e}")
            continue
    
    # 保存总对比结果
    comparison_summary = {
        'comparison_time': timestamp,
        'timesteps_tested': test_timesteps,
        'results': {
            int(ts): {
                'days': ts * 10,
                'test_f1': res['test_f1'],
                'test_acc': res['test_acc'],
                'test_f1_per_class': res['test_f1_per_class'],
                'save_dir': res['save_dir']
            }
            for ts, res in all_results.items()
        }
    }
    
    with open(f"{comparison_dir}/comparison_summary.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    
    # 可视化
    if len(all_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        timesteps_list = sorted(all_results.keys())
        f1_scores = [all_results[ts]['test_f1'] for ts in timesteps_list]
        accuracies = [all_results[ts]['test_acc'] for ts in timesteps_list]
        days_list = [ts * 10 for ts in timesteps_list]
        
        # F1对比
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(timesteps_list)), f1_scores, color='steelblue', alpha=0.8)
        ax1.set_xlabel('时间步 (天数)', fontsize=12)
        ax1.set_ylabel('F1 Score (macro)', fontsize=12)
        ax1.set_title('不同时间步的F1对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(timesteps_list)))
        ax1.set_xticklabels([f'{ts}步\n({days}天)' for ts, days in zip(timesteps_list, days_list)])
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, f1) in enumerate(zip(bars1, f1_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, f1 + 0.01,
                    f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 准确率对比
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(timesteps_list)), accuracies, color='coral', alpha=0.8)
        ax2.set_xlabel('时间步 (天数)', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_title('不同时间步的准确率对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(timesteps_list)))
        ax2.set_xticklabels([f'{ts}步\n({days}天)' for ts, days in zip(timesteps_list, days_list)])
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, acc) in enumerate(zip(bars2, accuracies)):
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{comparison_dir}/comparison_plot.png', dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存: {comparison_dir}/comparison_plot.png")
        
        # 总结
        print("\n" + "="*70)
        print("对比总结")
        print("="*70)
        best_ts = max(all_results.keys(), key=lambda ts: all_results[ts]['test_f1'])
        print(f"\n最佳时间步: {best_ts}步 ({best_ts*10}天)")
        print(f"  F1 Score: {all_results[best_ts]['test_f1']:.4f}")
        print(f"  准确率: {all_results[best_ts]['test_acc']:.4f}")
        print(f"  保存在: {all_results[best_ts]['save_dir']}")
        
        print("\n所有结果:")
        for ts in timesteps_list:
            print(f"  {ts}步({ts*10}天): F1={all_results[ts]['test_f1']:.4f}, "
                  f"Acc={all_results[ts]['test_acc']:.4f}")
        
        print(f"\n总对比目录: {comparison_dir}")
        print("="*70)

