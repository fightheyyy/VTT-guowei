"""
产量预测数据加载器
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class YieldDataset(Dataset):
    def __init__(self, csv_paths, selected_bands, input_steps=18, target_year=None):
        self.selected_bands = selected_bands
        self.input_steps = input_steps
        
        # 加载数据
        sequences, yields = [], []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            
            # 自动检测产量列（y2019, y2020, y2021, y2022）
            if target_year is None:
                # 查找CSV中存在的y开头的列
                yield_cols = [col for col in df.columns if col.startswith('y') and len(col) == 5]
                if len(yield_cols) == 0:
                    raise ValueError(f"在{csv_path}中未找到产量列（y2019, y2020等）")
                # 使用找到的第一个产量列
                actual_target = yield_cols[0]
            else:
                actual_target = target_year
            
            seq = []
            for band in selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(36)]
                seq.append(df[cols].values)
            sequences.append(np.stack(seq, axis=2))
            yields.append(df[actual_target].values)
        
        self.sequences = np.concatenate(sequences, axis=0)
        self.yields = np.concatenate(yields, axis=0)
        
        # 过滤掉产量为0或极小的样本
        valid_mask = self.yields > 0.1
        self.sequences = self.sequences[valid_mask]
        self.yields = self.yields[valid_mask]
        
        # 归一化
        self.seq_mean = self.sequences.mean(axis=(0, 1), keepdims=True)
        self.seq_std = self.sequences.std(axis=(0, 1), keepdims=True) + 1e-6
        self.sequences = (self.sequences - self.seq_mean) / self.seq_std
        
        self.yield_mean = self.yields.mean()
        self.yield_std = self.yields.std() + 1e-6
        self.yields = (self.yields - self.yield_mean) / self.yield_std
        
        print(f"  数据集: {len(self.sequences)} 样本, {input_steps}步({input_steps*10}天)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx, :self.input_steps, :]
        y = self.yields[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])


def create_yield_dataloaders(train_csv_paths, test_csv_paths, selected_bands, 
                             input_steps=18, batch_size=16):
    print(f"\n加载数据（输入={input_steps}步={input_steps*10}天）")
    
    train_dataset = YieldDataset(train_csv_paths, selected_bands, input_steps)
    test_dataset = YieldDataset(test_csv_paths, selected_bands, input_steps)
    
    test_dataset.seq_mean = train_dataset.seq_mean
    test_dataset.seq_std = train_dataset.seq_std
    test_dataset.yield_mean = train_dataset.yield_mean
    test_dataset.yield_std = train_dataset.yield_std
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(selected_bands)

