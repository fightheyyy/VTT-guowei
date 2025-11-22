"""
TimesCLIP分类任务训练脚本 - 改进版
整合早期分类优化策略，目标超越CLEC框架
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from experiments.classification.data_loader_classification_cached import create_classification_dataloaders_cached
from models.timesclip_classifier import TimesCLIPClassifier, LanguageOnlyTimesCLIPClassifier

# 导入改进的损失函数
from experiments.classification.improved_losses import (
    CombinedEarlyLoss,
    temporal_masking_augmentation,
    CurriculumScheduler
)

# 导入多时间尺度图像缓存
from experiments.classification.prepare_multiscale_images import MultiScaleImageCache

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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch_improved(model, train_loader, optimizer, device, 
                             criterion, curriculum_scheduler, 
                             epoch, total_epochs, use_dual_modal=True,
                             use_cached_images=False, multiscale_cache=None,
                             sample_indices=None):
    """
    改进的训练函数
    
    关键改进：
    1. 时间masking增强
    2. 时间感知的损失函数
    3. 课程学习策略
    4. 多时间尺度预生成图像（新增）
    
    Args:
        multiscale_cache: MultiScaleImageCache对象，用于加载预生成的不同时间长度的图像
        sample_indices: 样本索引列表，用于从multiscale_cache中加载正确的图像
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # 获取当前epoch的时间范围
    min_ratio, max_ratio = curriculum_scheduler.get_time_range(epoch)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}', leave=False)
    batch_start_idx = 0  # 跟踪当前batch在整个数据集中的起始索引
    
    for batch_data in pbar:
        # 处理数据（根据实际返回值自动判断）
        if len(batch_data) == 3:
            x, y, cached_images = batch_data
            x, y = x.to(device), y.to(device)
            cached_images = cached_images.to(device)
        else:
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            cached_images = None
        
        batch_size = x.size(0)
        
        # 【关键改进1】时间masking增强
        x_masked, keep_steps, time_ratio = temporal_masking_augmentation(
            x, min_ratio=min_ratio, max_ratio=max_ratio
        )
        
        # 【关键改进NEW】使用多时间尺度预生成图像
        if multiscale_cache is not None and sample_indices is not None:
            # 加载与keep_steps匹配的预生成图像
            batch_cached_images = []
            for i in range(batch_size):
                sample_idx = sample_indices[batch_start_idx + i]
                images = multiscale_cache.load_images(sample_idx, keep_steps)
                batch_cached_images.append(images)
            cached_images = torch.stack(batch_cached_images).to(device)  # [B, V, 3, 224, 224]
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_dual_modal:
            logits, contrastive_loss = model(x_masked, return_contrastive_loss=True, 
                                             cached_images=cached_images)  # 现在可以使用了！
            
            # 【关键改进2】使用时间感知的损失函数
            # CombinedEarlyLoss直接调用，返回(loss, loss_dict)
            loss, loss_dict = criterion(
                logits, y,
                features_visual=None,  # language_only不需要
                features_language=None,
                time_ratio=time_ratio
            )
            
            # 如果有对比学习损失，加上时间加权
            if contrastive_loss is not None and contrastive_loss.item() > 0:
                time_weight = 1.0 + 2.0 * (1.0 - time_ratio)
                loss = loss + 0.1 * contrastive_loss * time_weight
        else:
            # language_only模型
            logits = model(x_masked, cached_images=None)
            loss, loss_dict = criterion(
                logits, y,
                time_ratio=time_ratio
            )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'time': f'{int(time_ratio*37)}步'
        })
        
        # 记录预测
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        # 更新batch索引
        batch_start_idx += batch_size
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, f1


def evaluate_detailed(model, data_loader, device, use_dual_modal=True, use_cached_images=False):
    """详细评估（包含每个类别的F1）"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            # 根据实际返回值自动判断
            if len(batch_data) == 3:
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
    
    # 计算指标
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # 每个类别的F1
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # 混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': conf_matrix
    }


def evaluate_at_timesteps(model, data_loader, device, time_steps_list, 
                         use_dual_modal=True, use_cached_images=False):
    """
    评估模型在不同时间步数下的性能
    关键：测试早期识别能力
    
    Args:
        time_steps_list: 要测试的时间步列表，例如 [3, 6, 9, 12, 15, 18]
    
    Returns:
        dict: {time_steps: {'f1': xxx, 'acc': xxx, ...}}
    """
    from experiments.classification.data_loader_classification_cached import CachedImageClassificationDataset
    
    print("\n评估不同时间步的早期识别能力...")
    results = {}
    
    # 获取模型的 patch_length
    patch_length = model.language_preprocessor.patch_length
    print(f"模型 patch_length: {patch_length}")
    
    for time_steps in time_steps_list:
        # 跳过时间步数小于 patch_length 的测试点
        if time_steps < patch_length:
            print(f"\n  时间步 {time_steps} (对应 {time_steps*10} 天)... 跳过 (小于 patch_length={patch_length})")
            continue
            
        print(f"\n  时间步 {time_steps} (对应 {time_steps*10} 天)...")
        
        # 创建截断的数据集
        dataset = data_loader.dataset
        
        # 临时截断数据
        original_n_time_steps = dataset.n_time_steps
        original_data = dataset.data_normalized.copy()
        
        dataset.truncate_time_steps(time_steps)
        
        # 评估
        metrics = evaluate_detailed(model, data_loader, device, 
                                   use_dual_modal=use_dual_modal,
                                   use_cached_images=use_cached_images)
        
        results[time_steps] = {
            'f1_macro': metrics['f1_macro'],
            'accuracy': metrics['accuracy'],
            'f1_per_class': metrics['f1_per_class']
        }
        
        print(f"    F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # 恢复原始数据
        dataset.data_normalized = original_data
        dataset.n_time_steps = original_n_time_steps
    
    return results


def plot_early_recognition_curve(results_dict, save_path):
    """
    绘制早期识别曲线
    
    Args:
        results_dict: {time_steps: {'f1_macro': xxx, 'accuracy': xxx}}
    """
    time_steps = sorted(results_dict.keys())
    days = [t * 10 for t in time_steps]
    f1_scores = [results_dict[t]['f1_macro'] for t in time_steps]
    accuracies = [results_dict[t]['accuracy'] for t in time_steps]
    
    plt.figure(figsize=(12, 6))
    
    # F1曲线
    plt.subplot(1, 2, 1)
    plt.plot(days, f1_scores, 'o-', linewidth=2, markersize=8, label='F1 Score')
    plt.axhline(y=0.8, color='r', linestyle='--', label='目标阈值 (0.8)')
    plt.xlabel('时间 (天)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('早期识别性能 - F1分数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 标注每个点的值
    for i, (d, f1) in enumerate(zip(days, f1_scores)):
        plt.text(d, f1 + 0.02, f'{f1:.3f}', ha='center', fontsize=9)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(days, accuracies, 's-', linewidth=2, markersize=8, 
             color='green', label='Accuracy')
    plt.axhline(y=0.8, color='r', linestyle='--', label='目标阈值 (0.8)')
    plt.xlabel('时间 (天)', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('早期识别性能 - 准确率', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 标注每个点的值
    for i, (d, acc) in enumerate(zip(days, accuracies)):
        plt.text(d, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 找到首次超过0.8的时间点
    early_recognition_time = None
    for t, f1 in zip(time_steps, f1_scores):
        if f1 >= 0.8:
            early_recognition_time = t
            break
    
    return early_recognition_time


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


def train_timesclip_classifier_improved(
    csv_path="../../data/2018four.csv",
    time_steps=37,
    n_variates=14,
    model_type="dual",  # 支持dual和language_only
    batch_size=64,
    epochs=100,
    lr=1e-4,
    patience=15,
    device=None,
    use_cached_images=False,  # 改进策略下图像可选（使用multiscale_cache时无需此项）
    # 改进策略参数（专注早期识别）
    focal_alpha=0.25,
    focal_gamma=2.0,
    time_weight_factor=3.0,  # 提高到3.0，更重视早期
    warmup_ratio=0.15,  # 缩短warmup，尽快进入早期训练
    min_ratio_start=0.5,  # 从50%开始（约18步）
    min_ratio_end=0.08,  # 降到8%（约3步）
    focus_early=True,  # 专注早期识别
    # 多时间尺度图像缓存
    multiscale_cache_dir="../../data/multiscale_image_cache",  # 预生成的多时间尺度图像目录
    time_steps_list=None,  # 预生成的时间尺度列表（None时默认[3,6,9,12,15,18]）
    test_only=False  # 仅测试模式（加载已有模型，不训练）
):
    """
    训练TimesCLIP分类器 - 改进版
    
    支持模型：
    - dual: 双模态（视觉+语言）使用多时间尺度图像缓存
    - language_only: 纯语言模态
    
    核心改进：
    1. 时间感知的Focal Loss
    2. 时间masking数据增强
    3. 课程学习策略
    4. 多时间尺度图像预生成（针对dual模型）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(42)
    
    print("="*70)
    print("TimesCLIP 早期分类训练 - 改进版")
    print("="*70)
    print(f"数据文件: {csv_path}")
    print(f"模型类型: {model_type}")
    print(f"时间步数: {time_steps}")
    print(f"变量数: {n_variates}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    print(f"设备: {device}")
    print("\n改进策略:")
    print(f"  [√] 时间感知Focal Loss (weight_factor={time_weight_factor})")
    print(f"  [√] 时间masking增强")
    print(f"  [√] 课程学习 (warmup={int(epochs*warmup_ratio)}轮)")
    print(f"  [√] 时间范围: {min_ratio_start:.1f} -> {min_ratio_end:.1f}")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, num_classes = create_classification_dataloaders_cached(
        csv_path=csv_path,
        batch_size=batch_size,
        use_cache=True,
        load_to_memory=False  # 动态读取，省内存
    )
    
    print(f"类别数: {num_classes}")
    
    # 【新增】初始化多时间尺度图像缓存
    multiscale_cache = None
    train_indices = None
    if multiscale_cache_dir is not None:
        print("\n加载多时间尺度图像缓存...")
        if time_steps_list is None:
            time_steps_list = [3, 6, 9, 12, 15, 18]  # 默认6个时间尺度（早期+中期）
        try:
            multiscale_cache = MultiScaleImageCache(
                cache_dir=multiscale_cache_dir,
                time_steps_list=time_steps_list,
                n_variates=n_variates
            )
            # 获取训练集样本索引
            train_indices = train_loader.dataset.indices if hasattr(train_loader.dataset, 'indices') else None
            if train_indices is None:
                train_indices = list(range(len(train_loader.dataset)))
            print(f"  [√] 多时间尺度缓存已加载")
            print(f"  [√] 支持的时间尺度: {time_steps_list}")
            print(f"  [√] 训练样本数: {len(train_indices)}")
        except Exception as e:
            print(f"  [×] 加载失败: {e}")
            print(f"  → 将不使用预生成图像")
            multiscale_cache = None
    
    # 创建模型
    print("\n创建模型...")
    if model_type == "dual":
        model = TimesCLIPClassifier(
            time_steps=time_steps,
            n_variates=n_variates,
            num_classes=num_classes
        ).to(device)
        use_dual_modal = True
        print("  模型类型: 双模态 (视觉+语言)")
    elif model_type == "language_only":
        model = LanguageOnlyTimesCLIPClassifier(
            time_steps=time_steps,
            n_variates=n_variates,
            num_classes=num_classes
        ).to(device)
        use_dual_modal = False
        print("  模型类型: 纯语言")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 请选择 'dual' 或 'language_only'")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params/1e6:.2f}M")
    print(f"  可训练: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    
    # 【关键改进】创建改进的损失函数
    criterion = CombinedEarlyLoss(
        num_classes=num_classes,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        time_weight_factor=time_weight_factor,
        contrastive_weight=0.1 if use_dual_modal else 0.0  # dual模型使用对比学习
    )
    
    # 【关键改进】创建课程学习调度器
    curriculum_scheduler = CurriculumScheduler(
        total_epochs=epochs,
        warmup_epochs=int(epochs * warmup_ratio),
        min_ratio_start=min_ratio_start,
        min_ratio_end=min_ratio_end
    )
    
    print(f"\n课程学习: {curriculum_scheduler}")
    
    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # 关注F1提升
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 仅测试模式 - 跳过训练
    if test_only:
        print("\n" + "="*70)
        print("仅测试模式 - 跳过训练，直接加载已有模型")
        print("="*70)
    else:
        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"experiments/classification/timesclip_improved/logs/{model_type}_{timestamp}"
        writer = SummaryWriter(log_dir)
        
        # 训练循环
        print(f"\n开始训练...")
        best_val_f1 = 0.0
        best_metrics = None
        patience_counter = 0
        train_history = []
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc, train_f1 = train_one_epoch_improved(
                model, train_loader, optimizer, device,
                criterion, curriculum_scheduler,
                epoch, epochs, use_dual_modal=use_dual_modal,
                use_cached_images=use_cached_images,
                multiscale_cache=multiscale_cache,
                sample_indices=train_indices
            )
            
            # 验证
            val_metrics = evaluate_detailed(model, val_loader, device, 
                                           use_dual_modal=use_dual_modal,
                                           use_cached_images=use_cached_images)
            val_f1 = val_metrics['f1_macro']
            val_acc = val_metrics['accuracy']
            
            # 学习率调整
            scheduler.step(val_f1)
            
            # 记录
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('F1/train', train_f1, epoch)
            writer.add_scalar('F1/val', val_f1, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # 记录课程学习进度
            min_r, max_r = curriculum_scheduler.get_time_range(epoch)
            writer.add_scalar('Curriculum/min_ratio', min_r, epoch)
            writer.add_scalar('Curriculum/max_ratio', max_r, epoch)
            
            train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_f1': train_f1,
                'val_loss': val_metrics['loss'],
                'val_f1': val_f1,
                'val_acc': val_acc,
                'min_time_ratio': min_r,
                'max_time_ratio': max_r
            })
            
            # 打印
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}: "
                      f"Loss={train_loss:.4f}, F1={train_f1:.4f} | "
                      f"Val F1={val_f1:.4f}, Acc={val_acc:.4f} | "
                      f"Time=[{min_r:.2f}, {max_r:.2f}]")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_metrics = val_metrics
                patience_counter = 0
                
                # 保存模型
                os.makedirs("experiments/classification/timesclip_improved/checkpoints", exist_ok=True)
                save_path = f"experiments/classification/timesclip_improved/checkpoints/{model_type}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                }, save_path)
                
                print(f"  [√] 保存最佳模型 (Val F1={val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停于Epoch {epoch}，最佳Val F1={best_val_f1:.4f}")
                    break
    
    # 加载最佳模型并测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(f"experiments/classification/timesclip_improved/checkpoints/{model_type}_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_detailed(model, test_loader, device, 
                                    use_dual_modal=use_dual_modal,
                                    use_cached_images=use_cached_images)
    
    print("\n" + "="*70)
    print("测试集结果 (完整37步):")
    print("="*70)
    print(f"F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"召回率: {test_metrics['recall']:.4f}")
    print(f"每类F1: {[f'{f:.3f}' for f in test_metrics['f1_per_class']]}")
    print("="*70)
    
    # 【关键评估】测试6个关键时间尺度的早期识别能力
    if focus_early:
        print("\n" + "="*70)
        print("早期识别能力测试 (关键时间尺度)")
        print("="*70)
        
        # 定义要测试的6个时间尺度
        early_time_steps = time_steps_list if time_steps_list else [3, 6, 9, 12, 15, 18]
        
        # 评估每个时间尺度
        early_results = evaluate_at_timesteps(
            model, test_loader, device, early_time_steps,
            use_dual_modal=use_dual_modal,
            use_cached_images=use_cached_images
        )
        
        # 打印详细结果
        print("\n详细结果:")
        print("-" * 70)
        print(f"{'时间步':<8} {'天数':<8} {'F1 Score':<12} {'准确率':<12} {'是否达标'}")
        print("-" * 70)
        
        # 只遍历实际测试的时间步（跳过被过滤的）
        for t in sorted(early_results.keys()):
            days = t * 10
            f1 = early_results[t]['f1_macro']
            acc = early_results[t]['accuracy']
            达标 = "[√] 达标" if f1 >= 0.8 else "[×] 未达标"
            print(f"{t:<8} {days:<8} {f1:<12.4f} {acc:<12.4f} {达标}")
        
        print("-" * 70)
        
        # 找到首次达标的时间点
        early_recognition_time = None
        for t in sorted(early_results.keys()):
            if early_results[t]['f1_macro'] >= 0.8:
                early_recognition_time = t
                break
        
        if early_recognition_time:
            print(f"\n[目标] 最早可识别时间: {early_recognition_time}步 ({early_recognition_time*10}天)")
            print(f"   F1 Score: {early_results[early_recognition_time]['f1_macro']:.4f}")
        else:
            max_t = max(early_results.keys())
            print(f"\n[警告] 在{max_t}步({max_t*10}天)内未达到F1>0.8")
            print(f"   最佳表现: {max_t}步, F1={early_results[max_t]['f1_macro']:.4f}")
        
        print("="*70)
        
        # 绘制早期识别曲线
        curve_path = f"experiments/classification/timesclip_improved/results/{model_type}_early_recognition_curve.png"
        earliest_time = plot_early_recognition_curve(early_results, curve_path)
        print(f"\n早期识别曲线已保存: {curve_path}")
    
    # 保存结果
    os.makedirs("experiments/classification/timesclip_improved/results", exist_ok=True)
    
    # 绘制混淆矩阵
    conf_matrix_path = f"experiments/classification/timesclip_improved/results/{model_type}_confusion_matrix.png"
    plot_confusion_matrix(test_metrics['confusion_matrix'], conf_matrix_path)
    print(f"\n混淆矩阵已保存: {conf_matrix_path}")
    
    # 保存结果JSON
    results = {
        'model_type': model_type,
        'test_metrics': {
            'f1_macro': float(test_metrics['f1_macro']),
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1_per_class': test_metrics['f1_per_class'],
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        },
        'best_val_f1': float(best_val_f1),
        'training_history': train_history,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'time_weight_factor': time_weight_factor,
            'warmup_ratio': warmup_ratio,
            'min_ratio_start': min_ratio_start,
            'min_ratio_end': min_ratio_end,
            'focus_early': focus_early
        }
    }
    
    # 添加早期识别结果
    if focus_early and 'early_results' in locals():
        results['early_recognition'] = {
            'time_steps_tested': early_time_steps,
            'results_by_timestep': {
                int(t): {
                    'f1_macro': float(early_results[t]['f1_macro']),
                    'accuracy': float(early_results[t]['accuracy']),
                    'f1_per_class': early_results[t]['f1_per_class'],
                    'days': int(t * 10),
                    '达标': early_results[t]['f1_macro'] >= 0.8
                }
                for t in early_time_steps
            },
            'earliest_recognition_time': int(early_recognition_time) if early_recognition_time else None,
            'earliest_recognition_days': int(early_recognition_time * 10) if early_recognition_time else None
        }
    
    results_path = f"experiments/classification/timesclip_improved/results/{model_type}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存: {results_path}")
    
    writer.close()
    
    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TimesCLIP早期分类训练 - 改进版')
    parser.add_argument('--csv_path', type=str, default='../../data/2018four.csv', help='CSV数据文件路径')
    parser.add_argument('--model_type', type=str, default='dual', choices=['dual', 'language_only'], 
                       help='模型类型: dual(双模态) 或 language_only(纯语言)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--time_weight_factor', type=float, default=3.0, help='早期时间权重倍数（默认3.0）')
    parser.add_argument('--warmup_ratio', type=float, default=0.15, help='warmup比例（默认0.15）')
    parser.add_argument('--min_ratio_start', type=float, default=0.5, help='初始最小时间比例（默认0.5=18步）')
    parser.add_argument('--min_ratio_end', type=float, default=0.08, help='最终最小时间比例（默认0.08=3步）')
    parser.add_argument('--focus_early', type=bool, default=True, help='是否专注早期识别（默认True）')
    parser.add_argument('--multiscale_cache_dir', type=str, default='../../data/multiscale_image_cache', 
                       help='多时间尺度图像缓存目录（默认使用预生成图像加速训练）')
    parser.add_argument('--test_only', action='store_true', help='仅测试模式（加载已有模型，不重新训练）')
    
    args = parser.parse_args()
    
    # 训练模型
    model, metrics = train_timesclip_classifier_improved(
        csv_path=args.csv_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        time_weight_factor=args.time_weight_factor,
        warmup_ratio=args.warmup_ratio,
        min_ratio_start=args.min_ratio_start,
        min_ratio_end=args.min_ratio_end,
        focus_early=args.focus_early,
        multiscale_cache_dir=args.multiscale_cache_dir,
        test_only=args.test_only
    )

