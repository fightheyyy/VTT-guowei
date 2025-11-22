"""
改进版端到端Pipeline训练
预测+分类，专门处理类别不平衡
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os
from tqdm import tqdm

from models.timesclip_forecaster import TimesCLIPForecaster
from models.timesclip_classifier import TimesCLIPClassifier


class FocalLoss(nn.Module):
    """Focal Loss for classification"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
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
        return focal_loss


class TwoStageDataset(Dataset):
    """端到端数据集"""
    def __init__(self, data, labels, input_len=6, output_len=37):
        self.data = data
        self.labels = labels
        self.input_len = input_len
        self.output_len = output_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_input = self.data[idx, :self.input_len, :]
        x_full = self.data[idx, :self.output_len, :]
        y = self.labels[idx]
        
        return (
            torch.FloatTensor(x_input),
            torch.FloatTensor(x_full),
            torch.LongTensor([y])[0]
        )


class TwoStagePipeline(nn.Module):
    """两阶段Pipeline"""
    def __init__(self, forecaster, classifier):
        super().__init__()
        self.forecaster = forecaster
        self.classifier = classifier
    
    def forward(self, x_input, return_forecast=False):
        # 阶段1: 预测完整序列
        x_pred = self.forecaster(x_input)
        
        # 阶段2: 对预测序列分类
        logits = self.classifier(x_pred)
        
        if return_forecast:
            return logits, x_pred
        return logits


def load_data(csv_path, time_steps=37, n_variates=14):
    """加载数据"""
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
    
    return X_normalized, y


def compute_class_weights(y, device):
    """计算类别权重"""
    class_counts = np.bincount(y.astype(int))
    total = len(y)
    num_classes = len(class_counts)
    weights = total / (num_classes * class_counts)
    return torch.FloatTensor(weights).to(device)


def train_one_epoch(pipeline, train_loader, optimizer, criterion_cls, 
                    criterion_reg, device, epoch, lambda_reg=0.5):
    """训练一个epoch"""
    pipeline.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for x_input, x_full, y in pbar:
        x_input = x_input.to(device)
        x_full = x_full.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits, x_pred = pipeline(x_input, return_forecast=True)
        
        # 分类损失 (Focal Loss处理不平衡)
        cls_loss = criterion_cls(logits, y)
        
        # 回归损失 (预测序列的MSE)
        reg_loss = criterion_reg(x_pred, x_full)
        
        # 总损失
        loss = cls_loss + lambda_reg * reg_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        
        # 预测
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'reg': f'{reg_loss.item():.4f}'
        })
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'reg_loss': total_reg_loss / len(train_loader),
        'f1': f1,
        'accuracy': acc
    }


def evaluate(pipeline, val_loader, criterion_cls, criterion_reg, 
            device, lambda_reg=0.5):
    """验证"""
    pipeline.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_input, x_full, y in val_loader:
            x_input = x_input.to(device)
            x_full = x_full.to(device)
            y = y.to(device)
            
            logits, x_pred = pipeline(x_input, return_forecast=True)
            
            cls_loss = criterion_cls(logits, y)
            reg_loss = criterion_reg(x_pred, x_full)
            loss = cls_loss + lambda_reg * reg_loss
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(val_loader),
        'cls_loss': total_cls_loss / len(val_loader),
        'reg_loss': total_reg_loss / len(val_loader),
        'f1': f1,
        'accuracy': acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_pipeline_improved(
    csv_path='../../data/2018four.csv',
    input_len=6,
    output_len=37,
    n_variates=14,
    decoder_type='mlp',
    batch_size=32,
    epochs=100,
    lr=5e-5,
    patience=20,
    device=None,
    dropout=0.3,
    weight_decay=1e-3,
    focal_gamma=2.0,
    lambda_reg=0.5,
    checkpoint_path=None
):
    """
    改进版端到端训练
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("端到端Pipeline训练 [改进版 - 处理类别不平衡]")
    print("="*70)
    print(f"输入长度: {input_len}步")
    print(f"输出长度: {output_len}步")
    print(f"解码器: {decoder_type}")
    print(f"Batch size: {batch_size}")
    print(f"学习率: {lr}")
    print(f"Dropout: {dropout}")
    print(f"Focal Loss gamma: {focal_gamma}")
    print(f"回归损失权重: {lambda_reg}")
    print("="*70)
    
    # 加载数据
    X, y = load_data(csv_path, time_steps=output_len, n_variates=n_variates)
    
    # 划分数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n数据划分:")
    print(f"  训练: {len(X_train)}")
    print(f"  验证: {len(X_val)}")
    print(f"  测试: {len(X_test)}")
    
    # 类别分布
    print(f"\n训练集类别分布:")
    for i in range(4):
        count = (y_train == i).sum()
        print(f"  Class {i}: {count} ({count/len(y_train)*100:.2f}%)")
    
    # 计算类别权重
    class_weights = compute_class_weights(y_train, device)
    print(f"\n类别权重: {class_weights.cpu().numpy()}")
    
    # 创建数据集
    train_dataset = TwoStageDataset(X_train, y_train, input_len, output_len)
    val_dataset = TwoStageDataset(X_val, y_val, input_len, output_len)
    test_dataset = TwoStageDataset(X_test, y_test, input_len, output_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("\n创建模型...")
    forecaster = TimesCLIPForecaster(
        input_len=input_len,
        output_len=output_len,
        n_variates=n_variates,
        decoder_type=decoder_type,
        use_vision=False,
        use_language=True,
        patch_length=2,
        stride=1,
        dropout=dropout
    ).to(device)
    
    classifier = TimesCLIPClassifier(
        time_steps=output_len,
        n_variates=n_variates,
        num_classes=4,
        patch_length=4,
        stride=4,
        use_variate_selection=True,
        use_contrastive=True,
        dropout=dropout
    ).to(device)
    
    pipeline = TwoStagePipeline(forecaster, classifier).to(device)
    
    # 如果有checkpoint，加载
    start_epoch = 1
    best_val_f1 = 0.0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n从checkpoint恢复: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        print(f"  从Epoch {checkpoint['epoch']}恢复")
        print(f"  当前最佳Val F1: {best_val_f1:.4f}")
        print(f"  将从Epoch {start_epoch}继续训练")
    
    # 损失函数
    criterion_cls = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    criterion_reg = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(
        pipeline.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True
    )
    
    # 训练
    print("\n开始训练...")
    patience_counter = 0
    
    save_dir = "experiments/forecasting/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(start_epoch, epochs + 1):
        # 训练
        train_metrics = train_one_epoch(
            pipeline, train_loader, optimizer, 
            criterion_cls, criterion_reg, device, epoch, lambda_reg
        )
        
        # 验证
        val_metrics = evaluate(
            pipeline, val_loader, criterion_cls, 
            criterion_reg, device, lambda_reg
        )
        
        # 学习率调整
        scheduler.step(val_metrics['f1'])
        
        # 打印
        print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, "
              f"Val F1={val_metrics['f1']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': pipeline.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
            save_path = f"{save_dir}/pipeline_improved_in{input_len}.pth"
            torch.save(checkpoint, save_path)
            print(f"  [√] 保存最佳模型 (Val F1={val_metrics['f1']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: {patience} epochs没有改善")
                break
    
    # 测试
    print("\n在测试集上评估...")
    test_metrics = evaluate(
        pipeline, test_loader, criterion_cls, 
        criterion_reg, device, lambda_reg
    )
    
    print("\n"+"="*70)
    print("测试集结果")
    print("="*70)
    print(f"F1 Score (macro): {test_metrics['f1']:.4f}")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print("\n分类报告:")
    print(classification_report(
        test_metrics['labels'],
        test_metrics['predictions'],
        target_names=[f'Class {i}' for i in range(4)]
    ))
    
    return pipeline, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='../../data/2018four.csv')
    parser.add_argument('--input_len', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='继续训练的checkpoint路径')
    
    args = parser.parse_args()
    
    pipeline, metrics = train_pipeline_improved(
        csv_path=args.csv_path,
        input_len=args.input_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        checkpoint_path=args.checkpoint
    )

