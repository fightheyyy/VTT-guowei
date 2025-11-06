"""
实验2: 补全对比实验
目标: 对比 "先补全再回归" vs "直接回归" 的性能差异
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class DirectRegressor(nn.Module):
    """直接回归模型"""
    def __init__(self, input_length, n_variates=7, hidden_dim=128):
        super().__init__()
        self.encoder = nn.LSTM(n_variates, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return self.regressor(h[-1])


class TwoStageRegressor(nn.Module):
    """两阶段模型: 补全 + 回归"""
    def __init__(self, input_length, target_length=36, n_variates=7, hidden_dim=128):
        super().__init__()
        self.input_length = input_length
        self.target_length = target_length
        self.n_variates = n_variates
        
        # Stage1: 补全
        self.completion_encoder = nn.LSTM(n_variates, hidden_dim, 2, batch_first=True)
        self.completion_decoder = nn.Linear(hidden_dim, (target_length - input_length) * n_variates)
        
        # Stage2: 回归
        self.regression_encoder = nn.LSTM(n_variates, hidden_dim, 2, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        # 补全
        _, (h, _) = self.completion_encoder(x)
        pred_future = self.completion_decoder(h[-1])
        pred_future = pred_future.view(batch_size, self.target_length - self.input_length, self.n_variates)
        full_seq = torch.cat([x, pred_future], dim=1)
        
        # 回归
        _, (h, _) = self.regression_encoder(full_seq)
        return self.regressor(h[-1])


class YieldDataset(Dataset):
    def __init__(self, csv_paths, selected_bands, input_length):
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
        y = self.yields[idx]
        return torch.FloatTensor(x_input), torch.FloatTensor(y)


def train_model(model, train_loader, test_loader, epochs=50, lr=1e-3, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Val Loss={val_loss:.6f}")
    
    model.load_state_dict(best_state)
    return model


def evaluate(model, test_loader, yield_mean, yield_std, device='cuda'):
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.extend(model(x).cpu().numpy())
            targets.extend(y.numpy())
    
    preds = np.array(preds) * yield_std + yield_mean
    targets = np.array(targets) * yield_std + yield_mean
    
    return {
        'rmse': float(np.sqrt(mean_squared_error(targets, preds))),
        'mae': float(mean_absolute_error(targets, preds)),
        'r2': float(r2_score(targets, preds))
    }


def run_experiment():
    print("="*70)
    print("实验2: 补全 vs 不补全对比")
    print("="*70)
    
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = ["extract2022_20251010_165007.csv"]
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_lengths = [6, 12, 18, 24]
    results = {}
    
    os.makedirs('experiments/completion_comparison/results', exist_ok=True)
    
    for input_len in input_lengths:
        print(f"\n{'='*70}")
        print(f"输入长度: {input_len}个月 ({input_len*10}天)")
        print(f"{'='*70}")
        
        # 加载数据
        train_dataset = YieldDataset(train_files, selected_bands, input_len)
        test_dataset = YieldDataset(test_files, selected_bands, input_len)
        test_dataset.seq_mean = train_dataset.seq_mean
        test_dataset.seq_std = train_dataset.seq_std
        test_dataset.yield_mean = train_dataset.yield_mean
        test_dataset.yield_std = train_dataset.yield_std
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 方法1: 直接回归
        print("\n[1/2] 训练直接回归...")
        model_direct = DirectRegressor(input_len).to(device)
        model_direct = train_model(model_direct, train_loader, test_loader, epochs=50, device=device)
        results_direct = evaluate(model_direct, test_loader, train_dataset.yield_mean, train_dataset.yield_std, device)
        
        # 方法2: 两阶段
        print("\n[2/2] 训练两阶段...")
        model_twostage = TwoStageRegressor(input_len).to(device)
        model_twostage = train_model(model_twostage, train_loader, test_loader, epochs=50, device=device)
        results_twostage = evaluate(model_twostage, test_loader, train_dataset.yield_mean, train_dataset.yield_std, device)
        
        # 对比
        improvement = (results_twostage['rmse'] - results_direct['rmse']) / results_twostage['rmse'] * 100
        
        print(f"\n结果对比:")
        print(f"  {'方法':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
        print(f"  {'直接回归':<15} {results_direct['rmse']:>10.4f} {results_direct['mae']:>10.4f} {results_direct['r2']:>10.4f}")
        print(f"  {'两阶段':<15} {results_twostage['rmse']:>10.4f} {results_twostage['mae']:>10.4f} {results_twostage['r2']:>10.4f}")
        print(f"\n  直接法提升: {improvement:+.2f}%")
        
        results[input_len] = {
            'direct': results_direct,
            'twostage': results_twostage,
            'improvement': improvement
        }
    
    # 保存结果
    with open('experiments/completion_comparison/results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    visualize_results(results)
    
    return results


def visualize_results(results):
    input_lengths = sorted(results.keys())
    direct_rmse = [results[l]['direct']['rmse'] for l in input_lengths]
    twostage_rmse = [results[l]['twostage']['rmse'] for l in input_lengths]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(input_lengths))
    width = 0.35
    
    ax.bar(x - width/2, direct_rmse, width, label='直接回归', color='#2ecc71')
    ax.bar(x + width/2, twostage_rmse, width, label='两阶段（补全）', color='#e74c3c')
    
    ax.set_xlabel('输入长度（月）')
    ax.set_ylabel('RMSE')
    ax.set_title('补全 vs 不补全对比')
    ax.set_xticks(x)
    ax.set_xticklabels(input_lengths)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/completion_comparison/results/comparison.png', dpi=300)
    print("\n图表已保存: experiments/completion_comparison/results/comparison.png")


if __name__ == "__main__":
    results = run_experiment()
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)

