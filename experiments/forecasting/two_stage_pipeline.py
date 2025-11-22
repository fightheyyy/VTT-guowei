"""
两阶段早期识别流程
阶段1: 序列预测 - 从早期部分序列预测完整序列
阶段2: 序列分类 - 对预测的完整序列进行分类
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from models.timesclip_forecaster import TimesCLIPForecaster
from models.timesclip_classifier import TimesCLIPClassifier


class TwoStageDataset(Dataset):
    """两阶段数据集"""
    
    def __init__(self, data, labels, input_len=6, output_len=37):
        self.data = data
        self.labels = labels
        self.input_len = input_len
        self.output_len = output_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_input = self.data[idx, :self.input_len, :]  # 早期序列
        x_full = self.data[idx, :self.output_len, :]   # 完整序列
        y = self.labels[idx]
        
        return (
            torch.FloatTensor(x_input),
            torch.FloatTensor(x_full),
            torch.LongTensor([y])[0]
        )


class TwoStagePipeline(nn.Module):
    """
    两阶段流程
    Stage 1: Forecaster - 预测完整序列
    Stage 2: Classifier - 分类
    """
    
    def __init__(
        self,
        forecaster,
        classifier,
        freeze_forecaster=False,
        freeze_classifier=False
    ):
        super().__init__()
        
        self.forecaster = forecaster
        self.classifier = classifier
        
        # 冻结参数
        if freeze_forecaster:
            for param in self.forecaster.parameters():
                param.requires_grad = False
        
        if freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
    
    def forward(self, x_input, return_forecast=False):
        """
        Args:
            x_input: [batch, input_len, n_variates]
            return_forecast: 是否返回预测序列
        
        Returns:
            logits: [batch, num_classes]
            或 (logits, x_full_pred) 如果return_forecast=True
        """
        # Stage 1: 预测完整序列
        x_full_pred = self.forecaster(x_input)
        
        # Stage 2: 分类
        logits = self.classifier(x_full_pred)
        
        if return_forecast:
            return logits, x_full_pred
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


def evaluate_pipeline(model, data_loader, device):
    """评估两阶段流程"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_input, x_full, y in data_loader:
            x_input = x_input.to(device)
            y = y.to(device)
            
            logits = model(x_input)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'f1_macro': f1_macro,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_stage1_only(
    csv_path='../../data/2018four.csv',
    input_len=6,
    output_len=37,
    decoder_type='mlp',
    batch_size=64,
    epochs=50,
    lr=1e-4
):
    """
    阶段1训练：仅训练预测器
    """
    
    print("\n" + "="*70)
    print("阶段1: 训练序列预测器")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    X, y = load_data(csv_path, output_len, 14)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 创建数据集
    train_dataset = TwoStageDataset(X_train, y_train, input_len, output_len)
    val_dataset = TwoStageDataset(X_val, y_val, input_len, output_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建预测器
    forecaster = TimesCLIPForecaster(
        input_len=input_len,
        output_len=output_len,
        n_variates=14,
        decoder_type=decoder_type,
        use_vision=False,
        use_language=True,
        patch_length=2,
        stride=1
    ).to(device)
    
    # 训练
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(forecaster.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # 训练
        forecaster.train()
        train_loss = 0
        for x_input, x_full, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_input, x_full = x_input.to(device), x_full.to(device)
            
            x_pred = forecaster(x_input)
            loss = criterion(x_pred, x_full)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        forecaster.eval()
        val_loss = 0
        with torch.no_grad():
            for x_input, x_full, _ in val_loader:
                x_input, x_full = x_input.to(device), x_full.to(device)
                x_pred = forecaster(x_input)
                loss = criterion(x_pred, x_full)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("experiments/forecasting/checkpoints", exist_ok=True)
            save_path = f"experiments/forecasting/checkpoints/forecaster_stage1_in{input_len}.pth"
            torch.save(forecaster.state_dict(), save_path)
            print(f"  [√] 保存最佳预测器")
    
    return forecaster, save_path


def train_stage2_only(
    forecaster,
    csv_path='../../data/2018four.csv',
    input_len=6,
    output_len=37,
    batch_size=64,
    epochs=50,
    lr=1e-4
):
    """
    阶段2训练：冻结预测器，仅训练分类器
    """
    
    print("\n" + "="*70)
    print("阶段2: 训练分类器（预测器冻结）")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    X, y = load_data(csv_path, output_len, 14)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 创建数据集
    train_dataset = TwoStageDataset(X_train, y_train, input_len, output_len)
    val_dataset = TwoStageDataset(X_val, y_val, input_len, output_len)
    test_dataset = TwoStageDataset(X_test, y_test, input_len, output_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建分类器
    classifier = TimesCLIPClassifier(
        time_steps=output_len,
        n_variates=14,
        num_classes=4,
        patch_length=4,
        stride=4
    ).to(device)
    
    # 创建流程
    pipeline = TwoStagePipeline(
        forecaster=forecaster,
        classifier=classifier,
        freeze_forecaster=True,  # 冻结预测器
        freeze_classifier=False
    ).to(device)
    
    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, pipeline.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    best_val_f1 = 0
    
    for epoch in range(1, epochs + 1):
        # 训练
        pipeline.train()
        train_loss = 0
        for x_input, x_full, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_input, y = x_input.to(device), y.to(device)
            
            logits = pipeline(x_input)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        val_metrics = evaluate_pipeline(pipeline, val_loader, device)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
              f"Val F1={val_metrics['f1_macro']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            os.makedirs("experiments/forecasting/checkpoints", exist_ok=True)
            save_path = f"experiments/forecasting/checkpoints/pipeline_stage2_in{input_len}.pth"
            torch.save(pipeline.state_dict(), save_path)
            print(f"  [√] 保存最佳流程 (Val F1={best_val_f1:.4f})")
    
    # 加载最佳模型测试
    pipeline.load_state_dict(torch.load(save_path))
    test_metrics = evaluate_pipeline(pipeline, test_loader, device)
    
    print("\n" + "="*70)
    print("测试集结果:")
    print("="*70)
    print(f"F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print("="*70)
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(test_metrics['labels'], test_metrics['predictions']))
    
    return pipeline, test_metrics


def train_end_to_end(
    csv_path='../../data/2018four.csv',
    input_len=6,
    output_len=37,
    decoder_type='mlp',
    batch_size=64,
    epochs=100,
    lr=1e-4,
    alpha=0.3,  # 预测损失权重
    resume=False  # 是否从checkpoint恢复训练
):
    """
    端到端联合训练
    Loss = alpha * MSE(预测) + (1-alpha) * CE(分类)
    """
    
    print("\n" + "="*70)
    print("端到端联合训练")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    X, y = load_data(csv_path, output_len, 14)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 创建数据集
    train_dataset = TwoStageDataset(X_train, y_train, input_len, output_len)
    val_dataset = TwoStageDataset(X_val, y_val, input_len, output_len)
    test_dataset = TwoStageDataset(X_test, y_test, input_len, output_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    forecaster = TimesCLIPForecaster(
        input_len=input_len,
        output_len=output_len,
        n_variates=14,
        decoder_type=decoder_type,
        use_vision=False,
        use_language=True,
        patch_length=2,
        stride=1
    ).to(device)
    
    classifier = TimesCLIPClassifier(
        time_steps=output_len,
        n_variates=14,
        num_classes=4,
        patch_length=4,
        stride=4
    ).to(device)
    
    pipeline = TwoStagePipeline(forecaster, classifier).to(device)
    
    # 损失函数
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW(pipeline.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_f1 = 0
    start_epoch = 1
    
    # 从checkpoint恢复训练
    checkpoint_path = f"experiments/forecasting/checkpoints/pipeline_e2e_in{input_len}.pth"
    if resume and os.path.exists(checkpoint_path):
        print(f"\n从checkpoint恢复训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # 加载模型和优化器状态
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            pipeline.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['best_val_f1']
            print(f"  从Epoch {checkpoint['epoch']}恢复")
            print(f"  当前最佳Val F1: {best_val_f1:.4f}")
            print(f"  将从Epoch {start_epoch}继续训练")
        else:
            # 旧格式checkpoint，只有model_state_dict
            pipeline.load_state_dict(checkpoint)
            print(f"  加载模型权重成功（旧格式）")
    elif resume:
        print(f"\n警告: 未找到checkpoint {checkpoint_path}，从头开始训练")
    
    for epoch in range(start_epoch, epochs + 1):
        # 训练
        pipeline.train()
        total_loss = 0
        
        for x_input, x_full, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_input, x_full, y = x_input.to(device), x_full.to(device), y.to(device)
            
            # 前向传播
            logits, x_pred = pipeline(x_input, return_forecast=True)
            
            # 混合损失
            loss_forecast = mse_loss(x_pred, x_full)
            loss_classify = ce_loss(logits, y)
            loss = alpha * loss_forecast + (1 - alpha) * loss_classify
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 验证
        val_metrics = evaluate_pipeline(pipeline, val_loader, device)
        scheduler.step(val_metrics['f1_macro'])
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
              f"Val F1={val_metrics['f1_macro']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            os.makedirs("experiments/forecasting/checkpoints", exist_ok=True)
            save_path = f"experiments/forecasting/checkpoints/pipeline_e2e_in{input_len}.pth"
            
            # 保存完整checkpoint（用于恢复训练）
            torch.save({
                'epoch': epoch,
                'model_state_dict': pipeline.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'config': {
                    'input_len': input_len,
                    'output_len': output_len,
                    'decoder_type': decoder_type,
                    'alpha': alpha
                }
            }, save_path)
            print(f"  [√] 保存最佳模型 (Val F1={best_val_f1:.4f})")
    
    # 测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(save_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        print(f"  最佳模型来自Epoch {checkpoint['epoch']}, Val F1={checkpoint['best_val_f1']:.4f}")
    else:
        pipeline.load_state_dict(checkpoint)
    
    test_metrics = evaluate_pipeline(pipeline, test_loader, device)
    
    print("\n" + "="*70)
    print("端到端测试集结果:")
    print("="*70)
    print(f"F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print("="*70)
    
    return pipeline, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='两阶段早期识别流程')
    parser.add_argument('--mode', type=str, default='e2e', choices=['stage1', 'stage2', 'e2e'])
    parser.add_argument('--input_len', type=int, default=6)
    parser.add_argument('--decoder_type', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resume', action='store_true', help='从checkpoint恢复训练')
    
    args = parser.parse_args()
    
    if args.mode == 'stage1':
        # 仅训练预测器
        forecaster, _ = train_stage1_only(
            input_len=args.input_len,
            decoder_type=args.decoder_type,
            epochs=args.epochs
        )
    
    elif args.mode == 'stage2':
        # 先训练预测器，再训练分类器
        forecaster, forecaster_path = train_stage1_only(
            input_len=args.input_len,
            decoder_type=args.decoder_type,
            epochs=args.epochs
        )
        
        pipeline, metrics = train_stage2_only(
            forecaster=forecaster,
            input_len=args.input_len,
            epochs=args.epochs
        )
    
    elif args.mode == 'e2e':
        # 端到端联合训练
        pipeline, metrics = train_end_to_end(
            input_len=args.input_len,
            decoder_type=args.decoder_type,
            epochs=args.epochs*2,
            resume=args.resume
        )

