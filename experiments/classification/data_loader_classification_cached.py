"""
分类任务数据加载器 - 图像预缓存版本
在初始化时预先生成所有图像，训练时直接使用缓存，避免重复绘图
性能提升: 10-50倍
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.preprocessor import VisualPreprocessor
from tqdm import tqdm


class CachedImageClassificationDataset(Dataset):
    """时间序列分类数据集 - 图像预缓存版本"""
    
    def __init__(self, csv_path, indices=None, n_variates=14, n_time_steps=37, 
                 use_cache=True, cache_images=True):
        """
        Args:
            csv_path: CSV文件路径
            indices: 使用的样本索引（用于train/test split）
            n_variates: 变量数量（波段数）
            n_time_steps: 时间步数
            use_cache: 是否使用图像缓存（提速10-50倍）
            cache_images: 是否缓存图像到内存
        """
        self.n_variates = n_variates
        self.n_time_steps = n_time_steps
        self.use_cache = use_cache
        self.cache_images = cache_images
        
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
        
        # === 预生成图像缓存 ===
        self.cached_images = None
        if self.use_cache and self.cache_images:
            print(f"开始预生成图像缓存（共{len(self.data)}个样本，{n_variates}个变量）...")
            self._generate_image_cache()
    
    def _generate_image_cache(self):
        """预生成所有图像并缓存到内存"""
        device = torch.device('cpu')  # 在CPU上生成缓存
        visual_preprocessor = VisualPreprocessor(image_size=224).to(device)
        
        self.cached_images = []
        
        # 使用tqdm显示进度
        for i in tqdm(range(len(self.data)), desc="生成图像缓存"):
            x = torch.from_numpy(self.data_normalized[i]).float().unsqueeze(0)  # (1, T, V)
            x = x.to(device)
            
            with torch.no_grad():
                images = visual_preprocessor(x)  # (1, V, 3, 224, 224)
            
            # 保存到CPU内存
            self.cached_images.append(images.squeeze(0).cpu())  # (V, 3, 224, 224)
        
        print(f"✓ 图像缓存生成完成！")
        
        # 清理preprocessor
        del visual_preprocessor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回:
            x: (time_steps, n_variates) - 原始时间序列
            y: 标签
            images: (n_variates, 3, 224, 224) - 预生成的图像（如果启用缓存）
        """
        x = torch.from_numpy(self.data_normalized[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.use_cache and self.cached_images is not None:
            images = self.cached_images[idx]  # 直接使用缓存的图像
            return x, y, images
        else:
            return x, y


def create_classification_dataloaders_cached(
    csv_path,
    batch_size=32,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    num_workers=0,  # 改为0，因为图像已缓存，不需要多进程
    use_cache=True
):
    """
    创建训练、验证、测试数据加载器（分层采样，图像预缓存版本）
    
    Args:
        csv_path: CSV文件路径
        batch_size: 批次大小
        test_size: 测试集比例
        val_size: 从训练集中划分的验证集比例
        random_state: 随机种子
        num_workers: 数据加载线程数（缓存版本建议设为0）
        use_cache: 是否使用图像缓存
        
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
    
    # 创建数据集（会自动生成图像缓存）
    print("\n[1/3] 创建训练集...")
    train_dataset = CachedImageClassificationDataset(
        csv_path, indices=train_indices, use_cache=use_cache
    )
    
    print("\n[2/3] 创建验证集...")
    val_dataset = CachedImageClassificationDataset(
        csv_path, indices=val_indices, use_cache=use_cache
    )
    
    print("\n[3/3] 创建测试集...")
    test_dataset = CachedImageClassificationDataset(
        csv_path, indices=test_indices, use_cache=use_cache
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    
    print(f"\n{'='*70}")
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"类别数: {num_classes}")
    print(f"图像缓存: {'✓ 已启用' if use_cache else '✗ 未启用'}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader, num_classes


def collate_fn_cached(batch):
    """
    自定义collate函数，处理带缓存图像的batch
    """
    if len(batch[0]) == 3:  # (x, y, images)
        xs, ys, images = zip(*batch)
        return (
            torch.stack(xs),
            torch.stack(ys),
            torch.stack(images)
        )
    else:  # (x, y)
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)


if __name__ == "__main__":
    print("="*70)
    print("测试图像缓存数据加载器")
    print("="*70)
    
    # 测试数据加载
    train_loader, val_loader, test_loader, num_classes = create_classification_dataloaders_cached(
        csv_path="../../data/2018four.csv",
        batch_size=32,
        use_cache=True
    )
    
    # 查看一个batch
    batch = next(iter(train_loader))
    if len(batch) == 3:
        x, y, images = batch
        print(f"\nBatch示例（带图像缓存）:")
        print(f"x.shape: {x.shape}")  # (batch_size, time_steps, n_variates)
        print(f"y.shape: {y.shape}")  # (batch_size,)
        print(f"images.shape: {images.shape}")  # (batch_size, n_variates, 3, 224, 224)
        print(f"y: {y[:10]}")
    else:
        x, y = batch
        print(f"\nBatch示例（无图像缓存）:")
        print(f"x.shape: {x.shape}")
        print(f"y.shape: {y.shape}")
        print(f"y: {y[:10]}")
    
    print("\n✓ 数据加载器测试通过！")

