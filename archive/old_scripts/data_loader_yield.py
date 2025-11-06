"""
产量预测数据加载器
支持不同输入长度（不同天数）
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class YieldPredictionDataset(Dataset):
    """
    产量预测数据集
    每个样本：前N个时间步 → 产量
    """
    
    def __init__(
        self,
        csv_paths,
        selected_bands,
        input_steps=18,
        max_steps=36,
        target_year='y2022',
        normalize=True
    ):
        """
        参数:
            - csv_paths: CSV文件路径列表
            - selected_bands: 选择的波段
            - input_steps: 输入时间步数（如18步=180天）
            - max_steps: 数据集的最大时间步数
            - target_year: 目标产量年份
            - normalize: 是否归一化
        """
        self.selected_bands = selected_bands
        self.input_steps = input_steps
        self.max_steps = max_steps
        self.target_year = target_year
        self.normalize = normalize
        
        # 加载数据
        self.sequences, self.yields = self._load_data(csv_paths)
        
        # 归一化
        if normalize:
            self.seq_mean = self.sequences.mean(axis=(0, 1), keepdims=True)
            self.seq_std = self.sequences.std(axis=(0, 1), keepdims=True) + 1e-6
            self.sequences = (self.sequences - self.seq_mean) / self.seq_std
            
            self.yield_mean = self.yields.mean()
            self.yield_std = self.yields.std() + 1e-6
            self.yields = (self.yields - self.yield_mean) / self.yield_std
            
            print(f"  数据归一化:")
            print(f"    序列: mean={self.seq_mean.mean():.2f}, std={self.seq_std.mean():.2f}")
            print(f"    产量: mean={self.yield_mean:.2f}, std={self.yield_std:.2f}")
    
    def _load_data(self, csv_paths):
        """加载CSV数据"""
        all_sequences = []
        all_yields = []
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            
            # 提取时间序列
            sequences = []
            for band in self.selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(self.max_steps)]
                band_data = df[cols].values
                sequences.append(band_data)
            
            # [N_samples, Max_Steps, N_Variates]
            sequences = np.stack(sequences, axis=2)
            all_sequences.append(sequences)
            
            # 提取产量
            yields = df[self.target_year].values
            all_yields.append(yields)
        
        # 合并
        all_sequences = np.concatenate(all_sequences, axis=0)
        all_yields = np.concatenate(all_yields, axis=0)
        
        print(f"  加载数据: {len(all_sequences)} 样本")
        print(f"  时间步范围: {self.max_steps} (使用前{self.input_steps}步)")
        print(f"  变量数: {len(self.selected_bands)}")
        print(f"  产量范围: [{all_yields.min():.2f}, {all_yields.max():.2f}]")
        
        return all_sequences, all_yields
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # 只取前input_steps个时间步
        x = self.sequences[idx, :self.input_steps, :]  # [Input_Steps, N_Variates]
        y = self.yields[idx]  # 标量
        
        return torch.FloatTensor(x), torch.FloatTensor([y])


def create_yield_dataloaders(
    train_csv_paths,
    test_csv_paths,
    selected_bands,
    input_steps=18,
    batch_size=16,
    num_workers=0
):
    """
    创建产量预测数据加载器
    
    参数:
        - train_csv_paths: 训练集CSV路径
        - test_csv_paths: 测试集CSV路径
        - selected_bands: 波段列表
        - input_steps: 输入时间步数
        - batch_size: batch大小
    
    返回:
        - train_loader, test_loader, n_variates
    """
    print(f"\n{'='*70}")
    print(f"创建数据加载器（输入={input_steps}步={input_steps*10}天）")
    print(f"{'='*70}")
    
    # 训练集
    print("\n[训练集]")
    train_dataset = YieldPredictionDataset(
        csv_paths=train_csv_paths,
        selected_bands=selected_bands,
        input_steps=input_steps,
        normalize=True
    )
    
    # 测试集
    print("\n[测试集]")
    test_dataset = YieldPredictionDataset(
        csv_paths=test_csv_paths,
        selected_bands=selected_bands,
        input_steps=input_steps,
        normalize=True
    )
    
    # 使用训练集的归一化参数
    test_dataset.seq_mean = train_dataset.seq_mean
    test_dataset.seq_std = train_dataset.seq_std
    test_dataset.yield_mean = train_dataset.yield_mean
    test_dataset.yield_std = train_dataset.yield_std
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n训练批次: {len(train_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    return train_loader, test_loader, len(selected_bands)


if __name__ == "__main__":
    # 测试数据加载
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = [
        "extract2022_20251010_165007.csv"
    ]
    
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    
    # 测试不同输入长度
    for input_steps in [6, 12, 18, 24, 30, 36]:
        train_loader, test_loader, n_variates = create_yield_dataloaders(
            train_csv_paths=train_files,
            test_csv_paths=test_files,
            selected_bands=selected_bands,
            input_steps=input_steps,
            batch_size=16
        )
        
        # 测试一个batch
        x, y = next(iter(train_loader))
        print(f"\nBatch示例:")
        print(f"  X: {x.shape} ({input_steps}步 = {input_steps*10}天)")
        print(f"  Y: {y.shape}")
        print()

