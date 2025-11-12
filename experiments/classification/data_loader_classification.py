"""
分类任务数据加载器
用于处理2018four.csv数据集的时间序列分类
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ClassificationDataset(Dataset):
    """时间序列分类数据集"""
    
    def __init__(self, csv_path, indices=None, n_variates=14, n_time_steps=37):
        """
        Args:
            csv_path: CSV文件路径
            indices: 使用的样本索引（用于train/test split）
            n_variates: 变量数量（波段数）
            n_time_steps: 时间步数
        """
        self.n_variates = n_variates
        self.n_time_steps = n_time_steps
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
        # 如果指定了indices，则只取这些样本
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        
        # 提取特征和标签
        features = df.iloc[:, :-1].values  # (n_samples, 518)
        self.labels = df.iloc[:, -1].values  # (n_samples,)
        
        # 重塑为 (n_samples, n_time_steps, n_variates)
        self.data = features.reshape(-1, n_time_steps, n_variates)
        
        # 标准化（每个样本独立标准化）
        self.data_normalized = np.zeros_like(self.data, dtype=np.float32)
        for i in range(len(self.data)):
            # Instance Normalization
            mean = self.data[i].mean()
            std = self.data[i].std()
            if std > 0:
                self.data_normalized[i] = (self.data[i] - mean) / std
            else:
                self.data_normalized[i] = self.data[i] - mean
        
        print(f"数据集: {len(self.data)} 样本, {n_time_steps}步, {n_variates}变量")
        print(f"类别分布: {np.bincount(self.labels)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回 (time_steps, n_variates) 和标签
        x = torch.from_numpy(self.data_normalized[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_classification_dataloaders(
    csv_path,
    batch_size=32,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    num_workers=4
):
    """
    创建训练、验证、测试数据加载器（分层采样）
    
    Args:
        csv_path: CSV文件路径
        batch_size: 批次大小
        test_size: 测试集比例
        val_size: 从训练集中划分的验证集比例
        random_state: 随机种子
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # 先读取完整数据获取标签
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, -1].values
    indices = np.arange(len(labels))
    
    num_classes = len(np.unique(labels))
    
    # 分层划分训练集和测试集
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # 从训练集中再分层划分出验证集
    train_labels = labels[train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=random_state,
        stratify=train_labels
    )
    
    # 创建数据集
    train_dataset = ClassificationDataset(csv_path, indices=train_indices)
    val_dataset = ClassificationDataset(csv_path, indices=val_indices)
    test_dataset = ClassificationDataset(csv_path, indices=test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n数据集划分完成:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"类别数: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader, test_loader, num_classes = create_classification_dataloaders(
        csv_path="data/2018four.csv",
        batch_size=32
    )
    
    # 查看一个batch
    x, y = next(iter(train_loader))
    print(f"\nBatch示例:")
    print(f"x.shape: {x.shape}")  # (batch_size, time_steps, n_variates)
    print(f"y.shape: {y.shape}")  # (batch_size,)
    print(f"y: {y[:10]}")

