"""
多波段时间序列数据加载器
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MultiSpectralDataset(Dataset):
    """
    多光谱波段时间序列数据集
    """
    
    def __init__(self, csv_path, selected_bands=None, lookback=24, prediction_steps=12, 
                 train=True, test_size=0.2, random_state=42):
        """
        参数:
            csv_path: CSV文件路径
            selected_bands: 选择的波段列表，None表示使用所有波段
            lookback: 输入序列长度（从36个时间步中取前N个作为输入）
            prediction_steps: 预测步长
            train: 是否为训练集
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.csv_path = csv_path
        self.lookback = lookback
        self.prediction_steps = prediction_steps
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
        # 忽略y2019-y2021列，只保留波段数据
        # 所有可用的波段（每个波段有36个时间步，_00到_35）
        all_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'bsi', 'evi', 
                     'gcvi', 'green', 'lswi', 'ndsi', 'ndvi', 'ndwi', 
                     'ndyi', 'red']
        
        if selected_bands is None:
            selected_bands = all_bands
        
        self.selected_bands = selected_bands
        self.n_variates = len(selected_bands)
        
        print(f"使用的波段: {selected_bands}")
        
        # 提取每个波段的36个时间步数据
        data_list = []
        for idx in range(len(df)):
            sample_data = []
            for band in selected_bands:
                # 获取该波段的36个时间步列
                band_cols = [f'{band}_{i:02d}' for i in range(36)]
                band_values = df.loc[idx, band_cols].values.astype(np.float32)
                sample_data.append(band_values)
            
            # 转换为 [n_variates, 36] 的数组
            sample_data = np.array(sample_data)
            data_list.append(sample_data)
        
        # 转换为 [n_samples, n_variates, 36]
        self.data = np.array(data_list)
        
        # 划分训练集和测试集
        indices = np.arange(len(self.data))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        if train:
            self.data = self.data[train_idx]
        else:
            self.data = self.data[test_idx]
        
        print(f"{'训练集' if train else '测试集'} 样本数: {len(self.data)}")
        print(f"波段数: {self.n_variates}")
        print(f"每个波段时间步数: 36")
        print(f"输入序列长度: {lookback}")
        print(f"预测步长: {prediction_steps}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回:
            x: [lookback, n_variates] 输入序列
            y: [n_variates, prediction_steps] 目标序列
        """
        sample = self.data[idx]  # [n_variates, 36]
        
        # 输入：前lookback个时间步
        x = sample[:, :self.lookback].T  # [lookback, n_variates]
        
        # 目标：接下来的prediction_steps个时间步
        y = sample[:, self.lookback:self.lookback + self.prediction_steps]  # [n_variates, prediction_steps]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_dataloaders(csv_path, selected_bands=None, lookback=24, 
                       prediction_steps=12, batch_size=32, 
                       test_size=0.2, random_state=42, num_workers=0):
    """
    创建训练和测试数据加载器
    
    参数:
        csv_path: CSV文件路径
        selected_bands: 选择的波段列表
        lookback: 输入序列长度
        prediction_steps: 预测步长
        batch_size: 批次大小
        test_size: 测试集比例
        random_state: 随机种子
        num_workers: 数据加载线程数
    
    返回:
        train_loader, test_loader, n_variates
    """
    # 创建数据集
    train_dataset = MultiSpectralDataset(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=lookback,
        prediction_steps=prediction_steps,
        train=True,
        test_size=test_size,
        random_state=random_state
    )
    
    test_dataset = MultiSpectralDataset(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=lookback,
        prediction_steps=prediction_steps,
        train=False,
        test_size=test_size,
        random_state=random_state
    )
    
    # 创建数据加载器
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
    # 测试数据加载器
    csv_path = "extract2022_20251010_165007.csv"
    
    # 选择部分波段进行测试（可以根据需要调整）
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    
    train_loader, test_loader, n_variates = create_dataloaders(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=24,
        prediction_steps=12,
        batch_size=8
    )
    
    print(f"\n波段列表: {selected_bands}")
    print(f"波段数量: {n_variates}")
    
    # 测试加载一个批次
    for x, y in train_loader:
        print(f"\n输入批次形状: {x.shape}")  # [batch, lookback, n_variates]
        print(f"目标批次形状: {y.shape}")  # [batch, n_variates, prediction_steps]
        print(f"输入值范围: [{x.min():.2f}, {x.max():.2f}]")
        print(f"目标值范围: [{y.min():.2f}, {y.max():.2f}]")
        break
    
    print("\n数据加载器测试完成！")

