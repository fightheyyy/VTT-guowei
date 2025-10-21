"""
TimesCLIP模型训练脚本
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models import TimesCLIP
from data_loader import create_dataloaders


def train_one_epoch(model, train_loader, optimizer, device, lambda_gen=1.0, lambda_align=0.1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_contrastive = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc='训练中')
    for x, y in pbar:
        x = x.to(device)  # [batch, lookback, n_variates]
        y = y.to(device)  # [batch, n_variates, prediction_steps]
        
        # 前向传播
        y_pred, contrastive_loss = model(x, return_loss=True)
        
        # 计算总损失
        total_loss_batch, loss_dict = model.compute_loss(
            y_pred=y_pred,
            y_true=y,
            contrastive_loss=contrastive_loss,
            lambda_gen=lambda_gen,
            lambda_align=lambda_align
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累积损失
        total_loss += loss_dict['total_loss']
        total_mse += loss_dict['mse_loss']
        total_contrastive += loss_dict['contrastive_loss']
        n_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'mse': f"{loss_dict['mse_loss']:.4f}",
            'contrast': f"{loss_dict['contrastive_loss']:.4f}"
        })
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'contrastive': total_contrastive / n_batches
    }


def validate(model, test_loader, device, lambda_gen=1.0, lambda_align=0.1):
    """验证"""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_contrastive = 0
    n_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='验证中'):
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            y_pred, contrastive_loss = model(x, return_loss=True)
            
            # 计算损失
            total_loss_batch, loss_dict = model.compute_loss(
                y_pred=y_pred,
                y_true=y,
                contrastive_loss=contrastive_loss,
                lambda_gen=lambda_gen,
                lambda_align=lambda_align
            )
            
            total_loss += loss_dict['total_loss']
            total_mse += loss_dict['mse_loss']
            total_contrastive += loss_dict['contrastive_loss']
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'contrastive': total_contrastive / n_batches
    }


def train(
    csv_path,
    selected_bands=None,
    lookback=24,
    prediction_steps=12,
    batch_size=16,
    epochs=50,
    lr_vision=1e-5,
    lr_other=1e-4,
    lambda_gen=1.0,
    lambda_align=0.1,
    d_model=256,
    patch_length=8,
    stride=4,
    device='cuda',
    save_dir='checkpoints',
    log_dir='logs'
):
    """
    完整训练流程
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建数据加载器
    print("正在加载数据...")
    train_loader, test_loader, n_variates = create_dataloaders(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=lookback,
        prediction_steps=prediction_steps,
        batch_size=batch_size
    )
    
    # 初始化模型
    print("\n正在初始化模型...")
    model = TimesCLIP(
        time_steps=lookback,
        n_variates=n_variates,
        prediction_steps=prediction_steps,
        patch_length=patch_length,
        stride=stride,
        d_model=d_model
    )
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器
    param_groups = model.get_parameter_groups(
        lr_vision=lr_vision,
        lr_other=lr_other
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    best_val_loss = float('inf')
    
    print(f"\n开始训练 (共{epochs}个epoch)...")
    print("=" * 70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, lambda_gen, lambda_align
        )
        
        # 验证
        val_metrics = validate(
            model, test_loader, device, lambda_gen, lambda_align
        )
        
        # 更新学习率
        scheduler.step()
        
        # 打印指标
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, "
              f"MSE: {train_metrics['mse']:.4f}, "
              f"Contrast: {train_metrics['contrastive']:.4f}")
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, "
              f"MSE: {val_metrics['mse']:.4f}, "
              f"Contrast: {val_metrics['contrastive']:.4f}")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('MSE/train', train_metrics['mse'], epoch)
        writer.add_scalar('MSE/val', val_metrics['mse'], epoch)
        writer.add_scalar('Contrastive/train', train_metrics['contrastive'], epoch)
        writer.add_scalar('Contrastive/val', val_metrics['contrastive'], epoch)
        writer.add_scalar('LR/vision', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/other', optimizer.param_groups[1]['lr'], epoch)
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': {
                    'lookback': lookback,
                    'prediction_steps': prediction_steps,
                    'n_variates': n_variates,
                    'd_model': d_model,
                    'patch_length': patch_length,
                    'stride': stride
                }
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"保存最佳模型 (验证损失: {val_metrics['loss']:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    # 配置
    csv_path = "extract2022_20251010_165007.csv"
    
    # 选择部分波段（可根据需要调整）
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    
    # 训练
    train(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=24,           # 使用前24个时间步作为输入
        prediction_steps=12,   # 预测接下来12个时间步
        batch_size=16,
        epochs=50,
        lr_vision=1e-5,
        lr_other=1e-4,
        lambda_gen=1.0,
        lambda_align=0.1,
        d_model=256,           # 使用较小的模型以加快训练
        patch_length=8,
        stride=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

