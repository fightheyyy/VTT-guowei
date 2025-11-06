"""
使用多年数据训练的两阶段训练脚本（使用HF镜像）
训练集：2019-2021年
测试集：2022年
"""

import os

# 设置 HuggingFace 镜像（国内访问）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models import TimesCLIP
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
    train_loader, test_loader, n_variates = create_multiyear_dataloaders(
        train_csv_paths=train_csv_paths,
        test_csv_paths=test_csv_paths,
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
        patch_length=6,  # 对于lookback=18，使用较小的patch
        stride=3,
        d_model=d_model
    ).to(device)
    
    # 分层学习率
    vision_params = list(model.vision_module.parameters())
    other_params = [p for n, p in model.named_parameters() if 'vision_module' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': vision_params, 'lr': lr_vision},
        {'params': other_params, 'lr': lr_other}
    ])
    
    mse_loss = nn.MSELoss()
    writer = SummaryWriter('logs/stage1')
    
    best_loss = float('inf')
    
    print(f"\n训练参数:")
    print(f"  设备: {device}")
    print(f"  波段数: {n_variates}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  学习率 (Vision): {lr_vision}")
    print(f"  学习率 (Other): {lr_other}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_gen_loss = 0
        train_align_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (x, y_true) in enumerate(pbar):
            x = x.to(device)  # [B, lookback, n_variates]
            y_true = y_true.to(device)  # [B, n_variates, prediction_steps]
            
            optimizer.zero_grad()
            
            y_pred, align_loss = model(x)  # [B, n_variates, prediction_steps]
            
            gen_loss = mse_loss(y_pred, y_true)
            loss = lambda_gen * gen_loss + lambda_align * align_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            train_gen_loss += gen_loss.item()
            train_align_loss += align_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'gen': gen_loss.item(),
                'align': align_loss.item()
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_gen_loss = train_gen_loss / len(train_loader)
        avg_align_loss = train_align_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_true in test_loader:
                x = x.to(device)
                y_true = y_true.to(device)
                
                y_pred, align_loss = model(x)
                gen_loss = mse_loss(y_pred, y_true)
                loss = lambda_gen * gen_loss + lambda_align * align_loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f} (Gen: {avg_gen_loss:.6f}, Align: {avg_align_loss:.6f})')
        print(f'  Val Loss:   {avg_val_loss:.6f}')
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Loss/gen', avg_gen_loss, epoch)
        writer.add_scalar('Loss/align', avg_align_loss, epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'stage1_timeseries_best.pth'))
            print(f'  保存最佳模型 (Val Loss: {best_loss:.6f})')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(save_dir, f'stage1_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"\n阶段1完成! 最佳验证损失: {best_loss:.6f}")
    
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
    
    os.makedirs('logs/stage2', exist_ok=True)
    
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
    writer = SummaryWriter('logs/stage2')
    
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
            torch.save(yield_model.state_dict(), os.path.join(save_dir, 'stage2_yield_best.pth'))
            print(f'  保存最佳模型 (Val Loss: {best_loss:.6f}, R²: {r2:.4f})')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(yield_model.state_dict(), 
                      os.path.join(save_dir, f'stage2_epoch_{epoch+1}.pth'))
    
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
    
    # 阶段1: 时间序列补全
    timeseries_model = train_stage1_timeseries(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        lookback=18,
        prediction_steps=18,
        batch_size=8,  # 1080Ti显存较小，使用较小batch
        epochs=100,  # 增加训练轮数
        lr_vision=1e-5,  # 提高学习率加速收敛
        lr_other=1e-4,   # 提高学习率
        lambda_gen=1.0,
        lambda_align=0.2,  # 增加对齐损失权重
        d_model=256,
        device=device
    )
    
    # 阶段2: 产量预测（暂时注释，先对比阶段1）
    # yield_model = train_stage2_yield_prediction(
    #     train_csv_paths=train_files,
    #     test_csv_paths=test_files,
    #     timeseries_model=timeseries_model,
    #     selected_bands=selected_bands,
    #     batch_size=8,
    #     epochs=30,
    #     lr=1e-4,
    #     d_model=256,
    #     device=device
    # )
    
    print("\n" + "=" * 70)
    print("阶段1训练完成!")
    print("=" * 70)
    print(f"模型保存在: checkpoints/")
    print(f"  - stage1_timeseries_best.pth (时间序列补全)")
    # print(f"  - stage2_yield_best.pth (产量预测)")


if __name__ == "__main__":
    main()

