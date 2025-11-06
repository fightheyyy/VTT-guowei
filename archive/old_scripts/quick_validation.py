"""
快速验证实验
30分钟快速测试，验证实验设计是否合理
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

print("快速验证: 补全 vs 不补全")
print("="*60)

# 简化的模型
class SimpleDirectModel(nn.Module):
    def __init__(self, input_len, n_vars=7):
        super().__init__()
        self.lstm = nn.LSTM(n_vars, 64, 1, batch_first=True)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[0])


class SimpleTwoStageModel(nn.Module):
    def __init__(self, input_len, target_len, n_vars=7):
        super().__init__()
        self.input_len = input_len
        self.target_len = target_len
        
        # Stage1
        self.completion_lstm = nn.LSTM(n_vars, 64, 1, batch_first=True)
        self.completion_fc = nn.Linear(64, (target_len - input_len) * n_vars)
        
        # Stage2
        self.regression_lstm = nn.LSTM(n_vars, 64, 1, batch_first=True)
        self.regression_fc = nn.Linear(64, 1)
    
    def forward(self, x):
        # 补全
        _, (h, _) = self.completion_lstm(x)
        pred = self.completion_fc(h[0])
        pred = pred.view(-1, self.target_len - self.input_len, 7)
        full = torch.cat([x, pred], dim=1)
        
        # 回归
        _, (h, _) = self.regression_lstm(full)
        return self.regression_fc(h[0])


# 加载小数据集
print("\n加载数据...")
df = pd.read_csv("extract2022_20251010_165007.csv")

bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
sequences = []
for band in bands:
    cols = [f"{band}_{i:02d}" for i in range(36)]
    sequences.append(df[cols].values)

X = np.stack(sequences, axis=2)  # [N, 36, 7]
y = df['y2022'].values.reshape(-1, 1)

# 归一化
X = (X - X.mean()) / (X.std() + 1e-6)
y = (y - y.mean()) / (y.std() + 1e-6)

# 划分数据（使用100个样本快速测试）
X_train, y_train = X[:100], y[:100]
X_test, y_test = X[100:150], y[100:150]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}")
print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

# 测试不同输入长度
input_lengths = [6, 12, 18]
results = []

for input_len in input_lengths:
    print(f"\n{'='*60}")
    print(f"输入长度: {input_len}月")
    print(f"{'='*60}")
    
    # 准备数据
    X_train_input = torch.FloatTensor(X_train[:, :input_len, :])
    X_test_input = torch.FloatTensor(X_test[:, :input_len, :])
    y_train_t = torch.FloatTensor(y_train)
    y_test_t = torch.FloatTensor(y_test)
    
    # 方法1: 直接回归
    print("\n[1/2] 训练直接回归...")
    model_direct = SimpleDirectModel(input_len).to(device)
    optimizer = torch.optim.Adam(model_direct.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(20):  # 快速训练20轮
        model_direct.train()
        optimizer.zero_grad()
        pred = model_direct(X_train_input.to(device))
        loss = criterion(pred, y_train_t.to(device))
        loss.backward()
        optimizer.step()
    
    model_direct.eval()
    with torch.no_grad():
        pred_direct = model_direct(X_test_input.to(device)).cpu().numpy()
    
    rmse_direct = np.sqrt(mean_squared_error(y_test, pred_direct))
    r2_direct = r2_score(y_test, pred_direct)
    
    print(f"  RMSE: {rmse_direct:.4f}, R²: {r2_direct:.4f}")
    
    # 方法2: 两阶段
    print("\n[2/2] 训练两阶段...")
    model_twostage = SimpleTwoStageModel(input_len, 36).to(device)
    optimizer = torch.optim.Adam(model_twostage.parameters(), lr=1e-3)
    
    for epoch in range(20):
        model_twostage.train()
        optimizer.zero_grad()
        pred = model_twostage(X_train_input.to(device))
        loss = criterion(pred, y_train_t.to(device))
        loss.backward()
        optimizer.step()
    
    model_twostage.eval()
    with torch.no_grad():
        pred_twostage = model_twostage(X_test_input.to(device)).cpu().numpy()
    
    rmse_twostage = np.sqrt(mean_squared_error(y_test, pred_twostage))
    r2_twostage = r2_score(y_test, pred_twostage)
    
    print(f"  RMSE: {rmse_twostage:.4f}, R²: {r2_twostage:.4f}")
    
    # 对比
    improvement = (rmse_twostage - rmse_direct) / rmse_twostage * 100
    print(f"\n直接法相比两阶段: {improvement:+.1f}% RMSE")
    
    if improvement > 5:
        print("✅ 直接法更好")
    elif improvement < -5:
        print("❌ 两阶段更好")
    else:
        print("⚖ 性能接近")
    
    results.append({
        'input_length': input_len,
        'direct_rmse': rmse_direct,
        'twostage_rmse': rmse_twostage,
        'improvement': improvement
    })

# 总结
print(f"\n{'='*60}")
print("快速验证总结")
print(f"{'='*60}")

print(f"\n{'输入长度':<12} {'直接法RMSE':>12} {'两阶段RMSE':>12} {'提升':>10}")
print("-"*60)
for r in results:
    print(f"{r['input_length']:>3}月      {r['direct_rmse']:>12.4f} {r['twostage_rmse']:>12.4f} {r['improvement']:>9.1f}%")

avg_improvement = np.mean([r['improvement'] for r in results])
print(f"\n平均提升: {avg_improvement:+.1f}%")

if avg_improvement > 5:
    print("\n✅ 建议: 使用直接回归方法")
    print("   理由: 快速验证显示直接法性能更好")
elif avg_improvement < -5:
    print("\n⚠ 建议: 使用两阶段方法")
    print("   理由: 快速验证显示两阶段性能更好")
else:
    print("\n⚖ 建议: 两种方法都可考虑")
    print("   理由: 性能接近，需完整实验确认")

print("\n下一步:")
print("  python run_full_experiment.py  # 运行完整实验")
print("\n注意: 这只是快速验证（100样本，20轮），完整实验需要1500样本，50轮")

