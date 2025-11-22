"""
改进版TimesCLIP序列预测器训练
专门针对类别不平衡数据集的优化
"""

import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
from collections import Counter

from models.timesclip_forecaster import TimesCLIPForecaster


class FocalLoss(nn.Module):
    """
    Focal Loss - 专门处理类别不平衡
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重 [num_classes]
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImbalancedDatasetSampler(torch.utils.data.Sampler):
    """
    过采样少数类，平衡训练批次
    """
    def __init__(self, dataset, labels):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
        
        # 计算每个类别的权重
        label_counts = Counter(labels)
        weights = [1.0 / label_counts[labels[i]] for i in self.indices]
        self.weights = torch.DoubleTensor(weights)
    
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
    
    def __len__(self):
        return self.num_samples


class TimeSeriesAugmentation:
    """时间序列数据增强"""
    
    @staticmethod
    def add_noise(x, noise_level=0.01):
        """添加高斯噪声"""
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    @staticmethod
    def scale(x, scale_range=(0.9, 1.1)):
        """随机缩放"""
        scale = torch.FloatTensor(1).uniform_(*scale_range)
        return x * scale
    
    @staticmethod
    def time_shift(x, max_shift=1):
        """时间平移"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return x
        return torch.roll(x, shift, dims=0)
    
    @staticmethod
    def variate_dropout(x, dropout_prob=0.1):
        """随机丢弃变量"""
        mask = torch.rand(x.shape[-1]) > dropout_prob
        return x * mask.float()
    
    @staticmethod
    def augment(x, training=True):
        """组合增强"""
        if not training:
            return x
        
        # 随机应用增强
        if np.random.rand() < 0.5:
            x = TimeSeriesAugmentation.add_noise(x, noise_level=0.02)
        if np.random.rand() < 0.3:
            x = TimeSeriesAugmentation.scale(x, scale_range=(0.95, 1.05))
        if np.random.rand() < 0.3:
            x = TimeSeriesAugmentation.variate_dropout(x, dropout_prob=0.1)
        
        return x


class ForecastingDataset(Dataset):
    """改进的数据集，支持数据增强"""
    
    def __init__(self, data, input_len=6, output_len=37, augment=False):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_input = torch.FloatTensor(self.data[idx, :self.input_len, :])
        x_target = torch.FloatTensor(self.data[idx, :self.output_len, :])
        
        # 数据增强
        if self.augment:
            x_input = TimeSeriesAugmentation.augment(x_input, training=True)
            x_target = TimeSeriesAugmentation.augment(x_target, training=True)
        
        return x_input, x_target


def load_data(csv_path, time_steps=37, n_variates=14):
    """加载和预处理数据"""
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    n_samples = X.shape[0]
    X = X.reshape(n_samples, time_steps, n_variates)
    
    # 标准化
    X_normalized = np.zeros_like(X)
    for i in range(n_variates):
        variate_data = X[:, :, i]
        mean = variate_data.mean()
        std = variate_data.std() + 1e-8
        X_normalized[:, :, i] = (variate_data - mean) / std
    
    print(f"数据形状: {X_normalized.shape}")
    print(f"标签分布: {np.bincount(y.astype(int))}")
    
    return X_normalized, y


def compute_class_weights(y_train, device):
    """计算类别权重"""
    class_counts = np.bincount(y_train.astype(int))
    total = len(y_train)
    num_classes = len(class_counts)
    
    # 平衡权重
    weights = total / (num_classes * class_counts)
    weights = torch.FloatTensor(weights).to(device)
    
    print(f"\n类别权重:")
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.3f} (样本数: {class_counts[i]})")
    
    return weights


def train_one_epoch_improved(model, train_loader, optimizer, criterion, 
                            device, epoch, clip_grad_norm=1.0):
    """改进的训练循环"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (x_input, x_target) in enumerate(pbar):
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        x_pred = model(x_input)
        
        # 计算损失
        loss = criterion(x_pred, x_target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)


def evaluate_improved(model, val_loader, criterion, device):
    """改进的验证"""
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


def train_forecaster_improved(
    csv_path='../../data/2018four.csv',
    input_len=6,
    output_len=37,
    n_variates=14,
    decoder_type='mlp',
    use_vision=False,
    use_language=True,
    batch_size=32,  # 减小batch_size增加随机性
    epochs=100,
    lr=5e-5,  # 降低学习率
    patience=20,  # 增加patience
    device=None,
    # 改进策略开关
    use_focal_loss=True,
    use_data_augmentation=True,
    use_balanced_sampling=False,  # 平衡采样或类别权重二选一
    dropout=0.3,  # 增加dropout
    weight_decay=1e-3,  # 增加正则化
    focal_gamma=2.0,
    clip_grad_norm=1.0
):
    """
    改进版训练函数
    针对类别不平衡数据集优化
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("TimesCLIP序列预测器训练 [改进版]")
    print("="*70)
    print(f"输入长度: {input_len} 步")
    print(f"输出长度: {output_len} 步")
    print(f"预测长度: {output_len-input_len} 步")
    print(f"解码器类型: {decoder_type}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"Dropout: {dropout}")
    print(f"Weight Decay: {weight_decay}")
    print("\n改进策略:")
    print(f"  Focal Loss: {use_focal_loss}")
    print(f"  数据增强: {use_data_augmentation}")
    print(f"  平衡采样: {use_balanced_sampling}")
    print(f"  梯度裁剪: {clip_grad_norm}")
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
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 计算类别权重
    class_weights = compute_class_weights(y_train, device)
    
    # 创建数据集
    train_dataset = ForecastingDataset(
        X_train, input_len, output_len, 
        augment=use_data_augmentation
    )
    val_dataset = ForecastingDataset(X_val, input_len, output_len, augment=False)
    test_dataset = ForecastingDataset(X_test, input_len, output_len, augment=False)
    
    # 创建数据加载器
    if use_balanced_sampling:
        print("\n使用平衡采样器")
        train_sampler = ImbalancedDatasetSampler(train_dataset, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            sampler=train_sampler, num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=0
        )
    
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
        patch_length=2,
        stride=1,
        dropout=dropout  # 使用更高的dropout
    ).to(device)
    
    # 损失函数
    if use_focal_loss:
        print(f"\n使用Focal Loss (gamma={focal_gamma})")
        # 这里需要先转换为分类任务才能用Focal Loss
        # 对于回归任务，使用加权MSE
        criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True, min_lr=1e-6
    )
    
    # Cosine Annealing (额外的warmup)
    warmup_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # 创建保存目录
    save_dir = "experiments/forecasting/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_one_epoch_improved(
            model, train_loader, optimizer, criterion, 
            device, epoch, clip_grad_norm
        )
        
        # 验证
        val_loss = evaluate_improved(model, val_loader, criterion, device)
        
        # 学习率调整
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.6f}, "
              f"Val Loss={val_loss:.6f}, LR={current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            
            save_path = f"{save_dir}/forecaster_improved_best.pth"
            torch.save(checkpoint, save_path)
            print(f"  保存最佳模型 (Val Loss={val_loss:.6f})")
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n早停: {patience} epochs没有改善")
                break
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Improved)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/training_curve_improved.png', dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存: {save_dir}/training_curve_improved.png")
    
    # 测试
    print("\n在测试集上评估...")
    test_loss = evaluate_improved(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.6f}")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='../../data/2018four.csv')
    parser.add_argument('--input_len', type=int, default=6)
    parser.add_argument('--output_len', type=int, default=37)
    parser.add_argument('--decoder_type', type=str, default='mlp',
                       choices=['mlp', 'lstm', 'transformer'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--use_balanced_sampling', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    model, train_losses, val_losses = train_forecaster_improved(
        csv_path=args.csv_path,
        input_len=args.input_len,
        output_len=args.output_len,
        decoder_type=args.decoder_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        use_data_augmentation=args.use_augmentation,
        use_balanced_sampling=args.use_balanced_sampling,
        device=args.device
    )

