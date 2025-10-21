"""
支持产量预测的多波段时间序列数据加载器
两阶段训练：
  阶段1: 时间序列补全 (前N个时间步 → 后M个时间步)
  阶段2: 产量预测 (完整36步波段值 → 产量y)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MultiSpectralWithYieldDataset(Dataset):
    """
    多光谱波段+产量数据集
    支持两种模式：
      - 'timeseries': 时间序列预测（波段值补全）
      - 'yield': 产量预测（全年波段值→产量）
    """
    
    def __init__(self, csv_path, selected_bands=None, mode='timeseries',
                 lookback=18, prediction_steps=18, target_year='y2022',
                 train=True, test_size=0.2, random_state=42):
        """
        参数:
            csv_path: CSV文件路径
            selected_bands: 选择的波段列表
            mode: 'timeseries' 或 'yield'
            lookback: 时间序列模式下的输入长度（如1-5月约18步）
            prediction_steps: 时间序列模式下的预测长度（如6-12月约18步）
            target_year: 产量预测的目标年份列（y2019-y2022）
            train: 是否为训练集
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.csv_path = csv_path
        self.mode = mode
        self.lookback = lookback
        self.prediction_steps = prediction_steps
        self.target_year = target_year
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
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
        
        # 提取波段数据 [n_samples, n_variates, 36]
        data_list = []
        for idx in range(len(df)):
            sample_data = []
            for band in selected_bands:
                band_cols = [f'{band}_{i:02d}' for i in range(36)]
                band_values = df.loc[idx, band_cols].values.astype(np.float32)
                sample_data.append(band_values)
            sample_data = np.array(sample_data)
            data_list.append(sample_data)
        
        self.data = np.array(data_list)  # [n_samples, n_variates, 36]
        
        # 提取产量标签
        if target_year in df.columns:
            self.yields = df[target_year].values.astype(np.float32)
        else:
            print(f"警告: 列 '{target_year}' 不存在")
            self.yields = np.zeros(len(df), dtype=np.float32)
        
        # 划分训练/测试集
        indices = np.arange(len(self.data))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        if train:
            self.data = self.data[train_idx]
            self.yields = self.yields[train_idx]
        else:
            self.data = self.data[test_idx]
            self.yields = self.yields[test_idx]
        
        print(f"{'训练集' if train else '测试集'} 样本数: {len(self.data)}")
        
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


def create_dataloaders_with_yield(csv_path, selected_bands=None, mode='timeseries',
                                   lookback=18, prediction_steps=18, target_year='y2022',
                                   batch_size=32, test_size=0.2, random_state=42, 
                                   num_workers=0):
    """
    创建数据加载器（支持两种模式）
    """
    train_dataset = MultiSpectralWithYieldDataset(
        csv_path=csv_path,
        selected_bands=selected_bands,
        mode=mode,
        lookback=lookback,
        prediction_steps=prediction_steps,
        target_year=target_year,
        train=True,
        test_size=test_size,
        random_state=random_state
    )
    
    test_dataset = MultiSpectralWithYieldDataset(
        csv_path=csv_path,
        selected_bands=selected_bands,
        mode=mode,
        lookback=lookback,
        prediction_steps=prediction_steps,
        target_year=target_year,
        train=False,
        test_size=test_size,
        random_state=random_state
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
    csv_path = "extract2022_20251010_165007.csv"
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    
    print("=" * 70)
    print("测试模式1: 时间序列预测（波段值补全）")
    print("=" * 70)
    train_loader, test_loader, n_variates = create_dataloaders_with_yield(
        csv_path=csv_path,
        selected_bands=selected_bands,
        mode='timeseries',
        lookback=18,  # 1-5月（约18个时间步）
        prediction_steps=18,  # 6-12月（约18个时间步）
        batch_size=8
    )
    
    for x, y in train_loader:
        print(f"\n输入形状: {x.shape}  # [batch, lookback, n_variates]")
        print(f"目标形状: {y.shape}  # [batch, n_variates, prediction_steps]")
        break
    
    print("\n" + "=" * 70)
    print("测试模式2: 产量预测")
    print("=" * 70)
    train_loader, test_loader, n_variates = create_dataloaders_with_yield(
        csv_path=csv_path,
        selected_bands=selected_bands,
        mode='yield',
        target_year='y2022',
        batch_size=8
    )
    
    for x, y in train_loader:
        print(f"\n输入形状: {x.shape}  # [batch, 36, n_variates]")
        print(f"目标形状: {y.shape}  # [batch, 1] (产量)")
        print(f"产量值范围: [{y.min():.2f}, {y.max():.2f}]")
        break

