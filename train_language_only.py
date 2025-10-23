"""
消融实验：只使用语言模态训练
用于对比视觉模态的作用
"""

import os

# 设置 HuggingFace 镜像（国内访问）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models.timesclip_language_only import TimesCLIPLanguageOnly
from models.yield_predictor import YieldPredictor
from data_loader_multiyear import create_multiyear_dataloaders


def train_stage1_timeseries(
    train_csv_paths,
    test_csv_paths,
    selected_bands=None,
    lookback=18,
    prediction_steps=18,
    batch_size=16,
    epochs=50,
    lr_language=1e-5,
    lr_other=1e-4,
    d_model=256,
    device='cuda',
    save_dir='checkpoints'
):
    """
    阶段1: 训练时间序列补全模型（只用语言模态）
    """
    print("\n" + "=" * 70)
    print("阶段1: 训练波段值时间序列补全模型（语言模态版本）")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('logs/stage1_language_only', exist_ok=True)
    
    # 加载数据
    train_loader, test_loader, n_variates = create_multiyear_dataloaders(
        train_csv_paths=train_csv_paths,
        test_csv_paths=test_csv_paths,
        selected_bands=selected_bands,
        mode='timeseries',
        lookback=lookback,
        prediction_steps=prediction_steps,
        batch_size=batch_size
    )
    
    # 初始化模型（语言模态版本）
    model = TimesCLIPLanguageOnly(
        time_steps=lookback,
        n_variates=n_variates,
        prediction_steps=prediction_steps,
        patch_length=6,
        stride=3,
        d_model=d_model
    ).to(device)
    
    # 分层学习率
    language_params = list(model.language_module.parameters())
    other_params = [p for n, p in model.named_parameters() if 'language_module' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': language_params, 'lr': lr_language},
        {'params': other_params, 'lr': lr_other}
    ])
    
    mse_loss = nn.MSELoss()
    writer = SummaryWriter('logs/stage1_language_only')
    
    best_loss = float('inf')
    
    print(f"\n训练参数:")
    print(f"  设备: {device}")
    print(f"  波段数: {n_variates}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  学习率 (Language): {lr_language}")
    print(f"  学习率 (Other): {lr_other}")
    print(f"  模态: 仅语言模态（无视觉+对比学习）")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (x, y_true) in enumerate(pbar):
            x = x.to(device)  # [B, lookback, n_variates]
            y_true = y_true.to(device)  # [B, n_variates, prediction_steps]
            
            optimizer.zero_grad()
            
            # 前向传播（无对比损失）
            y_pred = model(x)  # [B, n_variates, prediction_steps]
            
            # 只有生成损失
            loss = mse_loss(y_pred, y_true)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_true in test_loader:
                x = x.to(device)
                y_true = y_true.to(device)
                
                y_pred = model(x)
                loss = mse_loss(y_pred, y_true)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f} (RMSE: {np.sqrt(avg_train_loss):.4f})')
        print(f'  Val Loss:   {avg_val_loss:.6f} (RMSE: {np.sqrt(avg_val_loss):.4f})')
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('RMSE/train', np.sqrt(avg_train_loss), epoch)
        writer.add_scalar('RMSE/val', np.sqrt(avg_val_loss), epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'stage1_language_only_best.pth'))
            print(f'  保存最佳模型 (Val Loss: {best_loss:.6f}, RMSE: {np.sqrt(best_loss):.4f})')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(save_dir, f'stage1_language_only_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"\n阶段1完成! 最佳验证损失: {best_loss:.6f} (RMSE: {np.sqrt(best_loss):.4f})")
    
    return model


def train_stage2_yield_prediction(
    train_csv_paths,
    test_csv_paths,
    timeseries_model,
    selected_bands=None,
    batch_size=16,
    epochs=30,
    lr=1e-4,
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
    
    os.makedirs('logs/stage2_language_only', exist_ok=True)
    
    # 加载产量预测数据
    train_loader, test_loader, n_variates = create_multiyear_dataloaders(
        train_csv_paths=train_csv_paths,
        test_csv_paths=test_csv_paths,
        selected_bands=selected_bands,
        mode='yield',
        batch_size=batch_size
    )
    
    # 初始化产量预测器（独立训练，不依赖阶段1）
    yield_model = YieldPredictor(
        n_variates=n_variates,
        time_steps=36,  # 全年数据
        d_model=d_model,
        n_heads=8,
        n_layers=4,
        dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.AdamW(yield_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    writer = SummaryWriter('logs/stage2_language_only')
    
    best_loss = float('inf')
    
    print(f"\n训练参数:")
    print(f"  设备: {device}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  学习率: {lr}")
    
    for epoch in range(epochs):
        yield_model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, y_true in pbar:
            x = x.to(device)  # [B, 36, n_variates]
            y_true = y_true.to(device)  # [B, 1]
            
            optimizer.zero_grad()
            
            # 直接预测产量
            y_pred = yield_model(x)  # [B, 1]
            
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        yield_model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y_true in test_loader:
                x = x.to(device)
                y_true = y_true.to(device)
                
                y_pred = yield_model(x)
                
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                
                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y_true.cpu().numpy())
        
        avg_val_loss = val_loss / len(test_loader)
        
        # 计算R²
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss:   {avg_val_loss:.6f}')
        print(f'  R² Score:   {r2:.4f}')
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/R2', r2, epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(yield_model.state_dict(), os.path.join(save_dir, 'stage2_language_only_best.pth'))
            print(f'  保存最佳模型 (Val Loss: {best_loss:.6f}, R²: {r2:.4f})')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(yield_model.state_dict(), 
                      os.path.join(save_dir, f'stage2_language_only_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"\n阶段2完成! 最佳验证损失: {best_loss:.6f}")
    
    return yield_model


def main():
    # 配置
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = [
        "extract2022_20251010_165007.csv"
    ]
    
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print("=" * 70)
    print("消融实验：语言模态版本（无视觉+对比学习）")
    print("=" * 70)
    
    # 阶段1: 时间序列补全
    timeseries_model = train_stage1_timeseries(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        lookback=18,
        prediction_steps=18,
        batch_size=8,
        epochs=100,
        lr_language=1e-5,
        lr_other=1e-4,
        d_model=256,
        device=device
    )
    
    # 阶段2: 产量预测
    yield_model = train_stage2_yield_prediction(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        timeseries_model=timeseries_model,
        selected_bands=selected_bands,
        batch_size=8,
        epochs=30,
        lr=1e-4,
        d_model=256,
        device=device
    )
    
    print("\n" + "=" * 70)
    print("两阶段训练全部完成!")
    print("=" * 70)
    print(f"模型保存在: checkpoints/")
    print(f"  - stage1_language_only_best.pth (时间序列补全)")
    print(f"  - stage2_language_only_best.pth (产量预测)")
    print(f"\n日志保存在: logs/")
    print(f"  - logs/stage1_language_only/")
    print(f"  - logs/stage2_language_only/")
    print("\n查看训练曲线: tensorboard --logdir=logs")


if __name__ == "__main__":
    main()

