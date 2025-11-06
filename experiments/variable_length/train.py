"""
实验3: 可变长度预测
目标: 任意前N个月预测剩余月份
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
from sklearn.metrics import mean_squared_error
import json


class VariableLengthModel(nn.Module):
    """可变长度预测模型"""
    def __init__(self, max_time_steps=36, n_variates=7, d_model=256):
        super().__init__()
        self.max_time_steps = max_time_steps
        
        self.encoder = nn.LSTM(n_variates, d_model, 2, batch_first=True)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_time_steps, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, d_model*4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        self.output_projection = nn.Linear(d_model, n_variates)
    
    def forward(self, x, input_length):
        batch_size = x.shape[0]
        prediction_len = self.max_time_steps - input_length
        
        # 编码
        _, (h, _) = self.encoder(x)
        memory = h.permute(1, 0, 2)  # [B, Layers, D]
        memory = memory.mean(dim=1, keepdim=True).expand(batch_size, input_length, -1)
        memory = memory + self.positional_encoding[:, :input_length, :]
        
        # 解码
        tgt = torch.zeros(batch_size, prediction_len, memory.shape[2], device=x.device)
        tgt = tgt + self.positional_encoding[:, input_length:input_length + prediction_len, :]
        
        decoded = self.decoder(tgt, memory)
        output = self.output_projection(decoded)
        
        return output


class VariableLengthDataset(Dataset):
    def __init__(self, csv_paths, selected_bands, fixed_input_length=None):
        sequences = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            seq = []
            for band in selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(36)]
                seq.append(df[cols].values)
            sequences.append(np.stack(seq, axis=2))
        
        self.sequences = np.concatenate(sequences, axis=0)
        self.fixed_input_length = fixed_input_length
        
        # 归一化
        self.mean = self.sequences.mean(axis=(0, 1), keepdims=True)
        self.std = self.sequences.std(axis=(0, 1), keepdims=True) + 1e-6
        self.sequences = (self.sequences - self.mean) / self.std
        
        print(f"  数据集: {len(self.sequences)} 样本")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        full_seq = self.sequences[idx]
        
        if self.fixed_input_length:
            input_len = self.fixed_input_length
        else:
            input_len = np.random.randint(3, 31)
        
        x = full_seq[:input_len]
        y = full_seq[input_len:]
        
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.tensor(input_len)


def run_experiment():
    print("="*70)
    print("实验3: 可变长度预测")
    print("="*70)
    
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = ["extract2022_20251010_165007.csv"]
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = VariableLengthModel(
        max_time_steps=36,
        n_variates=len(selected_bands),
        d_model=256
    ).to(device)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练数据（随机长度）
    train_dataset = VariableLengthDataset(train_files, selected_bands, fixed_input_length=None)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                              collate_fn=lambda b: (torch.nn.utils.rnn.pad_sequence([x[0] for x in b], batch_first=True),
                                                   torch.nn.utils.rnn.pad_sequence([x[1] for x in b], batch_first=True),
                                                   torch.stack([x[2] for x in b])))
    
    # 简化训练（仅展示框架）
    print("\n训练模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y, input_lengths in train_loader:
            x, y = x.to(device), y.to(device)
            
            # 简化：假设batch内长度相同
            input_len = input_lengths[0].item()
            
            optimizer.zero_grad()
            pred = model(x[:, :input_len], input_len)
            pred_len = 36 - input_len
            loss = criterion(pred[:, :pred_len], y[:, :pred_len])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.6f}")
    
    print("\n实验3框架已创建（需要完整实现训练和评估逻辑）")
    
    os.makedirs('experiments/variable_length/results', exist_ok=True)
    with open('experiments/variable_length/results/readme.txt', 'w') as f:
        f.write("可变长度预测实验\n")
        f.write("任意前N个月预测剩余月份\n")


if __name__ == "__main__":
    run_experiment()

