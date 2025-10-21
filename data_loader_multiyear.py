"""
支持多年数据合并的数据加载器
训练集：2019-2021年
测试集：2022年
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultiYearDataset(Dataset):
    """
    多年数据集
    支持两种模式：
      - 'timeseries': 时间序列预测（波段值补全）
      - 'yield': 产量预测（全年波段值→产量）
    """
    
    def __init__(self, csv_paths, selected_bands=None, mode='timeseries',
                 lookback=18, prediction_steps=18):
        """
        参数:
            csv_paths: CSV文件路径列表
            selected_bands: 选择的波段列表
            mode: 'timeseries' 或 'yield'
            lookback: 时间序列模式下的输入长度
            prediction_steps: 时间序列模式下的预测长度
        """
        self.mode = mode
        self.lookback = lookback
        self.prediction_steps = prediction_steps
        
        # 所有可用波段
        all_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'bsi', 'evi', 
                     'gcvi', 'green', 'lswi', 'ndsi', 'ndvi', 'ndwi', 
                     'ndyi', 'red']
        
        if selected_bands is None:
            selected_bands = all_bands
        
        self.selected_bands = selected_bands
        self.n_variates = len(selected_bands)
        
        print(f"使用的波段: {selected_bands}")
        print(f"模式: {mode}")
        
        # 合并多个CSV文件
        all_data = []
        all_yields = []
        
        for csv_path in csv_paths:
            print(f"加载: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 提取年份
            year = None
            for col in ['y2019', 'y2020', 'y2021', 'y2022']:
                if col in df.columns:
                    year = col
                    break
            
            # 提取波段数据
            for idx in range(len(df)):
                sample_data = []
                for band in selected_bands:
                    band_cols = [f'{band}_{i:02d}' for i in range(36)]
                    band_values = df.loc[idx, band_cols].values.astype(np.float32)
                    sample_data.append(band_values)
                sample_data = np.array(sample_data)
                all_data.append(sample_data)
                
                # 提取产量
                if year and year in df.columns:
                    all_yields.append(df.loc[idx, year])
                else:
                    all_yields.append(0.0)
        
        self.data = np.array(all_data)  # [n_samples, n_variates, 36]
        self.yields = np.array(all_yields, dtype=np.float32)
        
        print(f"总样本数: {len(self.data)}")
        print(f"产量范围: [{self.yields.min():.2f}, {self.yields.max():.2f}]")
        
        if mode == 'timeseries':
            print(f"时间序列模式: 前{lookback}步 → 后{prediction_steps}步")
        else:
            print(f"产量预测模式: 全年36步 → 产量y")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # [n_variates, 36]
        
        if self.mode == 'timeseries':
            # 时间序列预测模式
            x = sample[:, :self.lookback].T  # [lookback, n_variates]
            y = sample[:, self.lookback:self.lookback + self.prediction_steps]  # [n_variates, prediction_steps]
            return torch.FloatTensor(x), torch.FloatTensor(y)
        
        elif self.mode == 'yield':
            # 产量预测模式：使用全部36个时间步
            x = sample.T  # [36, n_variates]
            y = self.yields[idx]  # 标量产量值
            return torch.FloatTensor(x), torch.FloatTensor([y])
        
        else:
            raise ValueError(f"未知模式: {self.mode}")


def create_multiyear_dataloaders(train_csv_paths, test_csv_paths, 
                                  selected_bands=None, mode='timeseries',
                                  lookback=18, prediction_steps=18,
                                  batch_size=32, num_workers=0):
    """
    创建多年数据加载器
    
    参数:
        train_csv_paths: 训练集CSV文件路径列表（如2019-2021）
        test_csv_paths: 测试集CSV文件路径列表（如2022）
        selected_bands: 选择的波段列表
        mode: 'timeseries' 或 'yield'
        lookback: 时间序列模式下的输入长度
        prediction_steps: 时间序列模式下的预测长度
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    返回:
        train_loader, test_loader, n_variates
    """
    print("=" * 70)
    print("创建训练集")
    print("=" * 70)
    train_dataset = MultiYearDataset(
        csv_paths=train_csv_paths,
        selected_bands=selected_bands,
        mode=mode,
        lookback=lookback,
        prediction_steps=prediction_steps
    )
    
    print("\n" + "=" * 70)
    print("创建测试集")
    print("=" * 70)
    test_dataset = MultiYearDataset(
        csv_paths=test_csv_paths,
        selected_bands=selected_bands,
        mode=mode,
        lookback=lookback,
        prediction_steps=prediction_steps
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, test_loader, train_dataset.n_variates


if __name__ == "__main__":
    # 示例：使用2019-2021作为训练集，2022作为测试集
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = [
        "extract2022_20251010_165007.csv"
    ]
    
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    
    print("=" * 70)
    print("测试模式1: 时间序列预测（波段值补全）")
    print("=" * 70)
    train_loader, test_loader, n_variates = create_multiyear_dataloaders(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        mode='timeseries',
        lookback=18,
        prediction_steps=18,
        batch_size=8
    )
    
    print("\n训练集样本:")
    for x, y in train_loader:
        print(f"输入形状: {x.shape}  # [batch, lookback, n_variates]")
        print(f"目标形状: {y.shape}  # [batch, n_variates, prediction_steps]")
        break
    
    print("\n测试集样本:")
    for x, y in test_loader:
        print(f"输入形状: {x.shape}")
        print(f"目标形状: {y.shape}")
        break
    
    print("\n" + "=" * 70)
    print("测试模式2: 产量预测")
    print("=" * 70)
    train_loader, test_loader, n_variates = create_multiyear_dataloaders(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        mode='yield',
        batch_size=8
    )
    
    print("\n训练集样本:")
    for x, y in train_loader:
        print(f"输入形状: {x.shape}  # [batch, 36, n_variates]")
        print(f"目标形状: {y.shape}  # [batch, 1] (产量)")
        print(f"产量值范围: [{y.min():.2f}, {y.max():.2f}]")
        break
    
    print("\n测试集样本:")
    for x, y in test_loader:
        print(f"输入形状: {x.shape}")
        print(f"目标形状: {y.shape}")
        print(f"产量值范围: [{y.min():.2f}, {y.max():.2f}]")
        break

