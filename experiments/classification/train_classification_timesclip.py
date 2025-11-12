"""
TimesCLIP分类任务训练脚本
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from experiments.classification.data_loader_classification import create_classification_dataloaders
from experiments.classification.data_loader_classification_cached import create_classification_dataloaders_cached
from models.timesclip_classifier import TimesCLIPClassifier, LanguageOnlyTimesCLIPClassifier

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # 为了速度，关闭确定性
    torch.backends.cudnn.benchmark = True  # 开启benchmark以自动优化卷积算法


def train_one_epoch(model, train_loader, optimizer, device, contrastive_weight=0.1, use_dual_modal=True, epoch=1, total_epochs=100, use_cached_images=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}', leave=False)
    for batch_data in pbar:
        # 处理缓存图像
        if use_cached_images and len(batch_data) == 3:
            x, y, cached_images = batch_data
            x, y = x.to(device), y.to(device)
            cached_images = cached_images.to(device)
        else:
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            cached_images = None
        
        optimizer.zero_grad()
        
        if use_dual_modal:
            loss, loss_dict = model.compute_loss(x, y, contrastive_weight=contrastive_weight, cached_images=cached_images)
            # 直接从loss_dict获取logits，避免重复forward
            logits = loss_dict.get('logits', None)
        else:
            loss, loss_dict = model.compute_loss(x, y, cached_images=cached_images)
            logits = loss_dict.get('logits', None)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 预测（如果loss_dict中没有logits，需要重新forward）
        if logits is None:
            with torch.no_grad():
                if use_dual_modal:
                    logits = model(x, return_contrastive_loss=False, cached_images=cached_images)
                else:
                    logits = model(x, cached_images=cached_images)
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, data_loader, device, use_dual_modal=True, use_cached_images=False):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            # 处理缓存图像
            if use_cached_images and len(batch_data) == 3:
                x, y, cached_images = batch_data
                x, y = x.to(device), y.to(device)
                cached_images = cached_images.to(device)
            else:
                x, y = batch_data
                x, y = x.to(device), y.to(device)
                cached_images = None
            
            if use_dual_modal:
                logits = model(x, return_contrastive_loss=False, cached_images=cached_images)
            else:
                logits = model(x, cached_images=cached_images)
            
            loss = nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


def plot_confusion_matrix(conf_matrix, save_path, class_names=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f'类别{i}' for i in range(len(conf_matrix))]
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('混淆矩阵')
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_timesclip_classifier(
    csv_path="data/2018four.csv",
    time_steps=37,
    n_variates=14,
    model_type="dual",  # "dual" 或 "language_only"
    batch_size=32,
    epochs=100,
    lr=1e-4,
    contrastive_weight=0.1,
    patience=15,
    device=None,
    use_cached_images=True  # 使用图像缓存（提速10-50倍）
):
    """
    训练TimesCLIP分类器
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    set_seed(42)
    
    print("="*70)
    print("TimesCLIP 分类任务训练")
    print("="*70)
    print(f"数据文件: {csv_path}")
    print(f"模型类型: {'双模态' if model_type == 'dual' else '纯语言'}")
    print(f"时间步数: {time_steps}")
    print(f"变量数: {n_variates}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    if model_type == "dual":
        print(f"对比学习权重: {contrastive_weight}")
    print(f"图像缓存: {'✓ 启用' if use_cached_images else '✗ 禁用'}")
    print(f"设备: {device}")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    if use_cached_images:
        train_loader, val_loader, test_loader, num_classes = create_classification_dataloaders_cached(
            csv_path=csv_path,
            batch_size=batch_size,
            use_cache=True
        )
    else:
        train_loader, val_loader, test_loader, num_classes = create_classification_dataloaders(
            csv_path=csv_path,
            batch_size=batch_size
        )
    
    # 创建模型
    print("\n创建模型...")
    if model_type == "dual":
        model = TimesCLIPClassifier(
            time_steps=time_steps,
            n_variates=n_variates,
            num_classes=num_classes
        ).to(device)
        use_dual_modal = True
    else:
        model = LanguageOnlyTimesCLIPClassifier(
            time_steps=time_steps,
            n_variates=n_variates,
            num_classes=num_classes
        ).to(device)
        use_dual_modal = False
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params/1e6:.2f}M")
    print(f"  可训练: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    print(f"  冻结: {frozen_params/1e6:.2f}M")
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"experiments/classification/timesclip/logs/{model_type}_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    print(f"\n开始训练 {model_type}...")
    best_val_acc = 0.0
    patience_counter = 0
    train_history = []
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            contrastive_weight=contrastive_weight,
            use_dual_modal=use_dual_modal,
            epoch=epoch,
            total_epochs=epochs,
            use_cached_images=use_cached_images
        )
        
        # 验证
        val_metrics = evaluate(model, val_loader, device, use_dual_modal=use_dual_modal, use_cached_images=use_cached_images)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # 打印
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # 早停和保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存模型
            save_path = f"experiments/classification/timesclip/checkpoints/{model_type}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            
            print(f"  ✓ 保存最佳模型 (Val Acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停于Epoch {epoch}，最佳Val Acc={best_val_acc:.4f}")
                break
    
    # 加载最佳模型并测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(f"experiments/classification/timesclip/checkpoints/{model_type}_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, use_dual_modal=use_dual_modal, use_cached_images=use_cached_images)
    
    print("\n" + "="*70)
    print("测试集结果:")
    print("="*70)
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"召回率: {test_metrics['recall']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")
    print("="*70)
    
    # 绘制混淆矩阵
    conf_matrix_path = f"experiments/classification/timesclip/results/{model_type}_confusion_matrix.png"
    plot_confusion_matrix(test_metrics['confusion_matrix'], conf_matrix_path)
    print(f"\n混淆矩阵已保存: {conf_matrix_path}")
    
    # 保存结果
    results = {
        'model_type': model_type,
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        },
        'best_val_acc': float(best_val_acc),
        'train_history': train_history
    }
    
    results_path = f"experiments/classification/timesclip/results/{model_type}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存: {results_path}")
    
    writer.close()
    
    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TimesCLIP分类任务训练')
    parser.add_argument('--csv_path', type=str, default='data/2018four.csv', help='CSV数据文件路径')
    parser.add_argument('--model_type', type=str, default='dual', choices=['dual', 'language_only'], 
                        help='模型类型: dual(双模态) 或 language_only(纯语言)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小（默认64，开启缓存后可适当增大）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='对比学习权重')
    parser.add_argument('--no_cache', action='store_true', help='禁用图像缓存（不推荐，会慢10-50倍）')
    
    args = parser.parse_args()
    
    # 训练模型
    model, metrics = train_timesclip_classifier(
        csv_path=args.csv_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        contrastive_weight=args.contrastive_weight,
        use_cached_images=not args.no_cache  # 默认启用缓存
    )

