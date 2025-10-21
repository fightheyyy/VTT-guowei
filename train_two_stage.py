"""
两阶段训练脚本
阶段1: 训练TimesCLIP进行波段值时间序列补全
阶段2: 训练产量预测模型
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models import TimesCLIP
from models.yield_predictor import YieldPredictor
from data_loader_with_yield import create_dataloaders_with_yield


def train_stage1_timeseries(
    csv_path,
    selected_bands=None,
    lookback=18,
    prediction_steps=18,
    batch_size=16,
    epochs=50,
    lr_vision=1e-5,
    lr_other=1e-4,
    lambda_gen=1.0,
    lambda_align=0.1,
    d_model=256,
    device='cuda',
    save_dir='checkpoints'
):
    """
    阶段1: 训练时间序列补全模型
    """
    print("\n" + "=" * 70)
    print("阶段1: 训练波段值时间序列补全模型")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('logs/stage1', exist_ok=True)
    
    # 加载数据
    train_loader, test_loader, n_variates = create_dataloaders_with_yield(
        csv_path=csv_path,
        selected_bands=selected_bands,
        mode='timeseries',
        lookback=lookback,
        prediction_steps=prediction_steps,
        batch_size=batch_size
    )
    
    # 初始化模型
    model = TimesCLIP(
        time_steps=lookback,
        n_variates=n_variates,
        prediction_steps=prediction_steps,
        patch_length=8,
        stride=4,
        d_model=d_model
    ).to(device)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    param_groups = model.get_parameter_groups(lr_vision=lr_vision, lr_other=lr_other)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    writer = SummaryWriter('logs/stage1')
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x, y = x.to(device), y.to(device)
            
            y_pred, contrastive_loss = model(x, return_loss=True)
            total_loss, loss_dict = model.compute_loss(
                y_pred, y, contrastive_loss, lambda_gen, lambda_align
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss_dict['total_loss']
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred, contrastive_loss = model(x, return_loss=True)
                _, loss_dict = model.compute_loss(y_pred, y, contrastive_loss, lambda_gen, lambda_align)
                val_loss += loss_dict['total_loss']
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'lookback': lookback,
                    'prediction_steps': prediction_steps,
                    'n_variates': n_variates,
                    'd_model': d_model,
                    'patch_length': 8,
                    'stride': 4
                }
            }, os.path.join(save_dir, 'stage1_timeseries_best.pth'))
            print(f"保存最佳模型 (Val Loss: {val_loss:.4f})")
    
    writer.close()
    print(f"\n阶段1完成！最佳验证损失: {best_val_loss:.4f}")
    return model


def train_stage2_yield(
    csv_path,
    selected_bands=None,
    target_year='y2022',
    batch_size=32,
    epochs=100,
    lr=1e-3,
    d_model=256,
    device='cuda',
    save_dir='checkpoints'
):
    """
    阶段2: 训练产量预测模型
    """
    print("\n" + "=" * 70)
    print("阶段2: 训练产量预测模型")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('logs/stage2', exist_ok=True)
    
    # 加载数据
    train_loader, test_loader, n_variates = create_dataloaders_with_yield(
        csv_path=csv_path,
        selected_bands=selected_bands,
        mode='yield',
        target_year=target_year,
        batch_size=batch_size
    )
    
    # 初始化模型
    model = YieldPredictor(
        n_variates=n_variates,
        time_steps=36,
        d_model=d_model
    ).to(device)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    criterion = nn.MSELoss()
    writer = SummaryWriter('logs/stage2')
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(y_pred - y)).item()
        
        # 验证
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item()
                val_mae += torch.mean(torch.abs(y_pred - y)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        val_loss /= len(test_loader)
        val_mae /= len(test_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, MAE={train_mae:.4f} | "
              f"Val Loss={val_loss:.4f}, MAE={val_mae:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': {
                    'n_variates': n_variates,
                    'time_steps': 36,
                    'd_model': d_model
                }
            }, os.path.join(save_dir, 'stage2_yield_best.pth'))
            print(f"保存最佳模型 (Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f})")
    
    writer.close()
    print(f"\n阶段2完成！最佳验证损失: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    csv_path = "extract2022_20251010_165007.csv"
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 阶段1: 时间序列补全
    stage1_model = train_stage1_timeseries(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=18,           # 1-5月
        prediction_steps=18,   # 6-12月
        batch_size=8,          # 减小batch，增大模型
        epochs=80,             # 增加训练轮数
        lr_vision=1e-5,
        lr_other=5e-5,         # 降低学习率，更精细
        lambda_gen=1.0,
        lambda_align=0.01,     # 关键！从0.1降到0.01
        d_model=384,           # 增大模型容量
        device=device
    )
    
    # 阶段2: 产量预测
    stage2_model = train_stage2_yield(
        csv_path=csv_path,
        selected_bands=selected_bands,
        target_year='y2022',
        batch_size=16,         # 减小batch
        epochs=120,            # 稍微增加轮数
        lr=5e-4,              # 调整学习率
        d_model=384,           # 与阶段1匹配
        device=device
    )
    
    print("\n" + "=" * 70)
    print("两阶段训练全部完成！")
    print("=" * 70)
    print("\n模型文件:")
    print("  - checkpoints/stage1_timeseries_best.pth  (时间序列补全)")
    print("  - checkpoints/stage2_yield_best.pth       (产量预测)")
    print("\n可以使用 predict_2025.py 进行2025年预测")

