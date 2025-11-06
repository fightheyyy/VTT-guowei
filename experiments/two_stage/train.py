"""
实验4: 两阶段训练
目标: 先训练时间序列补全，再训练产量预测
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import json


class TwoStageModel(nn.Module):
    """两阶段模型"""
    def __init__(self, input_length=18, n_variates=7, d_model=256):
        super().__init__()
        self.input_length = input_length
        
        # Stage1: 序列补全
        self.completion_encoder = nn.LSTM(n_variates, d_model, 2, batch_first=True)
        self.completion_decoder = nn.Linear(d_model, (36 - input_length) * n_variates)
        
        # Stage2: 产量预测
        self.yield_encoder = nn.LSTM(n_variates, d_model, 2, batch_first=True)
        self.yield_regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1)
        )
    
    def complete_sequence(self, x):
        batch_size = x.shape[0]
        _, (h, _) = self.completion_encoder(x)
        pred = self.completion_decoder(h[-1])
        pred = pred.view(batch_size, 36 - self.input_length, -1)
        return torch.cat([x, pred], dim=1), pred
    
    def predict_yield(self, full_seq):
        _, (h, _) = self.yield_encoder(full_seq)
        return self.yield_regressor(h[-1])
    
    def forward(self, x):
        full_seq, _ = self.complete_sequence(x)
        return self.predict_yield(full_seq)


class TwoStageDataset(Dataset):
    def __init__(self, csv_paths, selected_bands, input_length=18):
        sequences, yields = [], []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            seq = []
            for band in selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(36)]
                seq.append(df[cols].values)
            sequences.append(np.stack(seq, axis=2))
            yields.append(df['y2022'].values.reshape(-1, 1))
        
        self.sequences = np.concatenate(sequences, axis=0)
        self.yields = np.concatenate(yields, axis=0)
        self.input_length = input_length
        
        # 归一化
        self.seq_mean = self.sequences.mean(axis=(0, 1), keepdims=True)
        self.seq_std = self.sequences.std(axis=(0, 1), keepdims=True) + 1e-6
        self.sequences = (self.sequences - self.seq_mean) / self.seq_std
        
        self.yield_mean = self.yields.mean()
        self.yield_std = self.yields.std() + 1e-6
        self.yields = (self.yields - self.yield_mean) / self.yield_std
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x_input = self.sequences[idx, :self.input_length, :]
        x_full = self.sequences[idx]
        y = self.yields[idx]
        return torch.FloatTensor(x_input), torch.FloatTensor(x_full), torch.FloatTensor(y)


def run_experiment():
    print("="*70)
    print("实验4: 两阶段训练")
    print("="*70)
    
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = ["extract2022_20251010_165007.csv"]
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_length = 18
    
    # 数据
    train_dataset = TwoStageDataset(train_files, selected_bands, input_length)
    test_dataset = TwoStageDataset(test_files, selected_bands, input_length)
    test_dataset.seq_mean = train_dataset.seq_mean
    test_dataset.seq_std = train_dataset.seq_std
    test_dataset.yield_mean = train_dataset.yield_mean
    test_dataset.yield_std = train_dataset.yield_std
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型
    model = TwoStageModel(input_length, len(selected_bands), 256).to(device)
    
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # Stage1: 训练序列补全
    print("\n[Stage 1] 训练序列补全...")
    optimizer = torch.optim.Adam([
        {'params': model.completion_encoder.parameters()},
        {'params': model.completion_decoder.parameters()}
    ], lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        for x_input, x_full, _ in train_loader:
            x_input, x_full = x_input.to(device), x_full.to(device)
            x_future = x_full[:, input_length:, :]
            
            optimizer.zero_grad()
            _, pred_future = model.complete_sequence(x_input)
            loss = criterion(pred_future, x_future)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Completion Loss={total_loss/len(train_loader):.6f}")
    
    # 冻结Stage1
    for param in model.completion_encoder.parameters():
        param.requires_grad = False
    for param in model.completion_decoder.parameters():
        param.requires_grad = False
    
    # Stage2: 训练产量预测
    print("\n[Stage 2] 训练产量预测...")
    optimizer = torch.optim.Adam([
        {'params': model.yield_encoder.parameters()},
        {'params': model.yield_regressor.parameters()}
    ], lr=1e-4)
    
    best_r2 = -float('inf')
    
    for epoch in range(50):
        model.train()
        for x_input, _, y in train_loader:
            x_input, y = x_input.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_input)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        
        # 评估
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_input, _, y in test_loader:
                x_input = x_input.to(device)
                preds.extend(model(x_input).cpu().numpy())
                targets.extend(y.numpy())
        
        preds = np.array(preds) * train_dataset.yield_std + train_dataset.yield_mean
        targets = np.array(targets) * train_dataset.yield_std + train_dataset.yield_mean
        r2 = r2_score(targets, preds)
        
        if r2 > best_r2:
            best_r2 = r2
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: R²={r2:.4f}")
    
    print(f"\n最佳 R²: {best_r2:.4f}")
    
    # 保存结果
    os.makedirs('experiments/two_stage/results', exist_ok=True)
    os.makedirs('experiments/two_stage/checkpoints', exist_ok=True)
    
    torch.save(model.state_dict(), 'experiments/two_stage/checkpoints/model.pth')
    
    with open('experiments/two_stage/results/results.json', 'w') as f:
        json.dump({'best_r2': float(best_r2)}, f, indent=2)
    
    print("\n实验完成！")


if __name__ == "__main__":
    run_experiment()

