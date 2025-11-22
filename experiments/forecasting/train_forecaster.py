"""
训练TimesCLIP序列预测器
任务：从早期部分序列预测完整序列
"""

import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm

from models.timesclip_forecaster import TimesCLIPForecaster


class ForecastingDataset(Dataset):
    """
    时间序列预测数据集
    输入: 前input_len步
    输出: 全部output_len步
    """
    
    def __init__(self, data, input_len=6, output_len=37):
        """
        Args:
            data: [n_samples, time_steps, n_variates]
            input_len: 输入序列长度
            output_len: 目标完整序列长度
        """
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        
        assert data.shape[1] >= output_len, f"数据时间步{data.shape[1]} < 目标长度{output_len}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 输入：前input_len步
        x_input = self.data[idx, :self.input_len, :]
        
        # 目标：完整output_len步
        x_target = self.data[idx, :self.output_len, :]
        
        return torch.FloatTensor(x_input), torch.FloatTensor(x_target)


def load_data(csv_path, time_steps=37, n_variates=14):
    """加载和预处理数据"""
    
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 分离特征和标签
    X = df.iloc[:, :-1].values  # 所有列除了最后的label
    y = df.iloc[:, -1].values   # label列
    
    # Reshape为3D: [n_samples, time_steps, n_variates]
    n_samples = X.shape[0]
    X = X.reshape(n_samples, time_steps, n_variates)
    
    # 标准化（按变量）
    X_normalized = np.zeros_like(X)
    for i in range(n_variates):
        variate_data = X[:, :, i]
        mean = variate_data.mean()
        std = variate_data.std() + 1e-8
        X_normalized[:, :, i] = (variate_data - mean) / std
    
    print(f"数据形状: {X_normalized.shape}")
    print(f"标签分布: {np.bincount(y.astype(int))}")
    
    return X_normalized, y


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for x_input, x_target in pbar:
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        
        # 前向传播
        x_pred = model(x_input)
        
        # 计算损失（预测部分）
        loss = criterion(x_pred, x_target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """评估模型"""
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_input, x_target in val_loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            
            x_pred = model(x_input)
            loss = criterion(x_pred, x_target)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def visualize_predictions(model, data, indices, input_len, device, save_path):
    """可视化预测结果"""
    
    model.eval()
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices[:9]):
            x_input = torch.FloatTensor(data[idx:idx+1, :input_len, :]).to(device)
            x_true = data[idx, :, :]
            
            x_pred = model(x_input).cpu().numpy()[0]
            
            # 随机选择一个变量绘制
            variate_idx = np.random.randint(0, data.shape[2])
            
            ax = axes[i]
            time_steps = np.arange(len(x_true))
            
            # 真实序列
            ax.plot(time_steps, x_true[:, variate_idx], 'b-', label='真实', linewidth=2)
            
            # 预测序列
            ax.plot(time_steps, x_pred[:, variate_idx], 'r--', label='预测', linewidth=2)
            
            # 输入部分标记
            ax.axvline(x=input_len-1, color='g', linestyle=':', linewidth=2, label='输入截断')
            
            ax.set_title(f'样本{idx} - 变量{variate_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存: {save_path}")


def train_forecaster(
    csv_path='../../data/2018four.csv',
    input_len=6,
    output_len=37,
    n_variates=14,
    decoder_type='mlp',
    use_vision=False,  # 预测任务可能不需要视觉
    use_language=True,
    batch_size=64,
    epochs=100,
    lr=1e-4,
    patience=15,
    device=None
):
    """
    训练TimesCLIP序列预测器
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("TimesCLIP序列预测器训练")
    print("="*70)
    print(f"输入长度: {input_len} 步 ({input_len*10} 天)")
    print(f"输出长度: {output_len} 步 ({output_len*10} 天)")
    print(f"预测长度: {output_len-input_len} 步")
    print(f"解码器类型: {decoder_type}")
    print(f"使用视觉: {use_vision}")
    print(f"使用语言: {use_language}")
    print(f"设备: {device}")
    print("="*70)
    
    # 加载数据
    X, y = load_data(csv_path, time_steps=output_len, n_variates=n_variates)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 创建数据集和加载器
    train_dataset = ForecastingDataset(X_train, input_len, output_len)
    val_dataset = ForecastingDataset(X_val, input_len, output_len)
    test_dataset = ForecastingDataset(X_test, input_len, output_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    print("\n创建模型...")
    model = TimesCLIPForecaster(
        input_len=input_len,
        output_len=output_len,
        n_variates=n_variates,
        decoder_type=decoder_type,
        use_vision=use_vision,
        use_language=use_language,
        patch_length=2,  # 短序列用小patch
        stride=1
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建保存目录
    os.makedirs("experiments/forecasting/checkpoints", exist_ok=True)
    os.makedirs("experiments/forecasting/results", exist_ok=True)
    
    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_path = f"experiments/forecasting/checkpoints/forecaster_{decoder_type}_in{input_len}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'input_len': input_len,
                    'output_len': output_len,
                    'n_variates': n_variates,
                    'decoder_type': decoder_type,
                    'use_vision': use_vision,
                    'use_language': use_language
                }
            }, save_path)
            print(f"  [√] 保存最佳模型 (Val Loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停于Epoch {epoch}，最佳Val Loss={best_val_loss:.6f}")
                break
    
    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*70)
    print("测试集结果:")
    print("="*70)
    print(f"MSE Loss: {test_loss:.6f}")
    print(f"RMSE: {np.sqrt(test_loss):.6f}")
    print("="*70)
    
    # 可视化预测
    print("\n生成可视化...")
    vis_indices = np.random.choice(len(X_test), size=9, replace=False)
    vis_path = f"experiments/forecasting/results/predictions_{decoder_type}_in{input_len}.png"
    visualize_predictions(model, X_test, vis_indices, input_len, device, vis_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training Curve - {decoder_type} Decoder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_curve_path = f"experiments/forecasting/results/loss_curve_{decoder_type}_in{input_len}.png"
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {loss_curve_path}")
    
    return model, test_loss


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练TimesCLIP序列预测器')
    parser.add_argument('--csv_path', type=str, default='../../data/2018four.csv')
    parser.add_argument('--input_len', type=int, default=6, help='输入序列长度')
    parser.add_argument('--output_len', type=int, default=37, help='输出序列长度')
    parser.add_argument('--decoder_type', type=str, default='mlp', choices=['mlp', 'lstm', 'transformer'])
    parser.add_argument('--use_vision', action='store_true', help='是否使用视觉分支')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # 训练模型
    model, test_loss = train_forecaster(
        csv_path=args.csv_path,
        input_len=args.input_len,
        output_len=args.output_len,
        decoder_type=args.decoder_type,
        use_vision=args.use_vision,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

