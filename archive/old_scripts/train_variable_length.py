"""
可变长度训练脚本
训练模型适应任意前N个月预测剩余月份
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models.timesclip_variable_length import TimesCLIPVariableLength
from data_loader_variable_length import create_variable_length_dataloaders


def train_variable_length(
    train_csv_paths,
    test_csv_paths,
    selected_bands,
    max_time_steps=36,
    min_input_length=3,
    max_input_length=30,
    test_input_lengths=[3, 6, 12, 18, 24],
    batch_size=16,
    epochs=50,
    lr=1e-4,
    d_model=256,
    device='cuda',
    save_dir='checkpoints'
):
    """
    训练可变长度预测模型
    """
    print("\n" + "=" * 70)
    print("训练可变长度时间序列预测模型")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('logs/variable_length', exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    train_loader, test_loader, n_variates = create_variable_length_dataloaders(
        train_csv_paths=train_csv_paths,
        test_csv_paths=test_csv_paths,
        selected_bands=selected_bands,
        max_time_steps=max_time_steps,
        min_input_length=min_input_length,
        max_input_length=max_input_length,
        test_input_length=18,  # 测试时默认用18个月
        batch_size=batch_size
    )
    
    # 初始化模型
    print("\n初始化模型...")
    model = TimesCLIPVariableLength(
        max_time_steps=max_time_steps,
        n_variates=n_variates,
        patch_length=6,
        stride=3,
        d_model=d_model
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    writer = SummaryWriter('logs/variable_length')
    
    best_loss = float('inf')
    
    print(f"\n训练参数:")
    print(f"  设备: {device}")
    print(f"  波段数: {n_variates}")
    print(f"  最大时间步: {max_time_steps}")
    print(f"  输入长度范围: {min_input_length}-{max_input_length}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  学习率: {lr}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    
    # 训练循环
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (x, y_true, input_lengths) in enumerate(pbar):
            # x: [Batch, Max_Input_Len, N_Variates] (padded)
            # y_true: [Batch, Max_Pred_Len, N_Variates] (padded)
            # input_lengths: [Batch]
            
            x = x.to(device)
            y_true = y_true.to(device)
            
            # 由于batch内输入长度相同（collate后padding），
            # 我们需要对每个样本单独处理
            batch_loss = 0
            batch_size_actual = x.shape[0]
            
            for i in range(batch_size_actual):
                input_len = input_lengths[i].item()
                
                # 提取实际数据（去除padding）
                x_i = x[i:i+1, :input_len, :]  # [1, Input_Len, N_Variates]
                pred_len = max_time_steps - input_len
                y_true_i = y_true[i:i+1, :pred_len, :]  # [1, Pred_Len, N_Variates]
                
                # 前向传播
                y_pred_i = model(x_i, input_length=input_len)
                
                # 计算损失
                loss_i = criterion(y_pred_i, y_true_i)
                batch_loss += loss_i
            
            # 平均损失
            loss = batch_loss / batch_size_actual
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x, y_true, input_lengths in test_loader:
                x = x.to(device)
                y_true = y_true.to(device)
                
                batch_loss = 0
                batch_size_actual = x.shape[0]
                
                for i in range(batch_size_actual):
                    input_len = input_lengths[i].item()
                    x_i = x[i:i+1, :input_len, :]
                    pred_len = max_time_steps - input_len
                    y_true_i = y_true[i:i+1, :pred_len, :]
                    
                    y_pred_i = model(x_i, input_length=input_len)
                    loss_i = criterion(y_pred_i, y_true_i)
                    batch_loss += loss_i
                
                loss = batch_loss / batch_size_actual
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        # 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, 'variable_length_best.pth')
            )
            print(f"  ✓ 保存最佳模型 (loss={val_loss:.6f})")
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'variable_length_epoch_{epoch+1}.pth')
            )
    
    writer.close()
    
    # 测试不同输入长度
    print("\n" + "=" * 70)
    print("测试不同输入长度的预测性能")
    print("=" * 70)
    
    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'variable_length_best.pth'))
    )
    model.eval()
    
    for test_len in test_input_lengths:
        # 创建测试数据加载器
        _, test_loader_len, _ = create_variable_length_dataloaders(
            train_csv_paths=train_csv_paths,
            test_csv_paths=test_csv_paths,
            selected_bands=selected_bands,
            max_time_steps=max_time_steps,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
            test_input_length=test_len,
            batch_size=batch_size
        )
        
        test_loss = 0
        with torch.no_grad():
            for x, y_true, input_lengths in test_loader_len:
                x = x.to(device)
                y_true = y_true.to(device)
                
                batch_loss = 0
                batch_size_actual = x.shape[0]
                
                for i in range(batch_size_actual):
                    input_len = input_lengths[i].item()
                    x_i = x[i:i+1, :input_len, :]
                    pred_len = max_time_steps - input_len
                    y_true_i = y_true[i:i+1, :pred_len, :]
                    
                    y_pred_i = model(x_i, input_length=input_len)
                    loss_i = criterion(y_pred_i, y_true_i)
                    batch_loss += loss_i
                
                loss = batch_loss / batch_size_actual
                test_loss += loss.item()
        
        test_loss /= len(test_loader_len)
        rmse = np.sqrt(test_loss)
        
        print(f"\n前{test_len}个月 → 预测后{max_time_steps - test_len}个月:")
        print(f"  MSE:  {test_loss:.6f}")
        print(f"  RMSE: {rmse:.4f}")
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"最佳模型保存在: {os.path.join(save_dir, 'variable_length_best.pth')}")
    print(f"查看训练曲线: tensorboard --logdir=logs/variable_length")


if __name__ == "__main__":
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
    
    # 训练
    train_variable_length(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        max_time_steps=36,
        min_input_length=3,    # 最少3个月
        max_input_length=30,   # 最多30个月
        test_input_lengths=[3, 6, 12, 18, 24],  # 测试这些长度
        batch_size=16,
        epochs=50,
        lr=1e-4,
        d_model=256,
        device=device
    )

