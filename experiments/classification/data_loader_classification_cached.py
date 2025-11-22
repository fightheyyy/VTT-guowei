"""
分类任务数据加载器 - 图像预缓存版本
在初始化时预先生成所有图像，训练时直接使用缓存，避免重复绘图
支持磁盘缓存，避免每次运行都重新生成
性能提升: 10-50倍
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
import os
import hashlib
import pickle
from PIL import Image
import torchvision.transforms as transforms
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.preprocessor import VisualPreprocessor
from tqdm import tqdm


class CachedImageClassificationDataset(Dataset):
    """时间序列分类数据集 - 图像预缓存版本"""
    
    def __init__(self, csv_path, indices=None, n_variates=14, n_time_steps=37, 
                 use_cache=True, cache_images=True, disk_cache_dir="../../data/image_cache",
                 load_to_memory=False):
        """
        Args:
            csv_path: CSV文件路径
            indices: 使用的样本索引（用于train/test split）
            n_variates: 变量数量（波段数）
            n_time_steps: 时间步数
            use_cache: 是否使用图像缓存（提速10-50倍）
            cache_images: 是否缓存图像到内存
            disk_cache_dir: 磁盘缓存目录（None则只使用内存缓存）
            load_to_memory: 是否将所有图像加载到内存（False则训练时动态从磁盘读取，省内存）
        """
        self.n_variates = n_variates
        self.n_time_steps = n_time_steps
        self.use_cache = use_cache
        self.cache_images = cache_images
        self.disk_cache_dir = disk_cache_dir
        self.csv_path = csv_path
        self.indices = indices
        self.load_to_memory = load_to_memory
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
        # 如果指定了indices，则只取这些样本
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        
        # 提取特征和标签
        features = df.iloc[:, :-1].values  # (n_samples, 518)
        self.labels = df.iloc[:, -1].values  # (n_samples,)
        
        # CSV包含完整的37步数据，先reshape成37步
        full_time_steps = 37
        n_samples = features.shape[0]
        expected_features = full_time_steps * n_variates
        
        if features.shape[1] != expected_features:
            raise ValueError(f"特征列数不匹配: 期望 {expected_features} (37×{n_variates}), 实际 {features.shape[1]}")
        
        # 重塑为完整的37步
        full_data = features.reshape(n_samples, full_time_steps, n_variates)
        
        # 如果需要的时间步数小于37，则截取前n_time_steps步
        if n_time_steps <= full_time_steps:
            self.data = full_data[:, :n_time_steps, :]  # 截取前n_time_steps步
        else:
            raise ValueError(f"请求的时间步数 {n_time_steps} 超过数据最大步数 {full_time_steps}")
        
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
            self._generate_image_cache()
    
    def _get_cache_dir(self):
        """获取缓存目录（支持扁平结构和哈希子目录）"""
        # 首先检查是否是扁平结构（图像直接在disk_cache_dir下）
        if self.disk_cache_dir and os.path.exists(self.disk_cache_dir):
            # 检查是否有sample_0_variate_0.png这样的文件（扁平结构）
            test_file = os.path.join(self.disk_cache_dir, "sample_0_variate_0.png")
            if os.path.exists(test_file):
                return self.disk_cache_dir  # 使用扁平结构
        
        # 否则使用哈希子目录结构
        cache_key = f"{os.path.abspath(self.csv_path)}_{str(sorted(self.indices) if self.indices is not None else 'all')}"
        hash_value = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        return os.path.join(self.disk_cache_dir, f"cache_{hash_value}")
    
    def _load_from_disk_cache(self):
        """从磁盘加载PNG图像缓存"""
        if self.disk_cache_dir is None:
            return False
        
        cache_dir = self._get_cache_dir()
        
        if not os.path.exists(cache_dir):
            return False
        
        # 检查是否是扁平结构
        is_flat_structure = (cache_dir == self.disk_cache_dir)
        
        if not is_flat_structure:
            # 哈希子目录结构：检查文件数
            expected_files = len(self.data) * self.n_variates
            existing_files = len([f for f in os.listdir(cache_dir) if f.endswith('.png')])
            
            if existing_files != expected_files:
                print(f"缓存文件不完整 ({existing_files}/{expected_files})")
                return False
        
        print(f"从磁盘加载图像缓存: {cache_dir}")
        if is_flat_structure:
            print(f"检测到扁平结构，将根据indices映射样本")
        
        # 定义图像转换（PNG -> Tensor）
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.cached_images = []
        
        try:
            for i in tqdm(range(len(self.data)), desc="从磁盘加载图像"):
                # 如果使用了indices，需要映射到原始样本编号
                if self.indices is not None:
                    original_idx = self.indices[i]
                else:
                    original_idx = i
                
                sample_images = []
                for v in range(self.n_variates):
                    img_path = os.path.join(cache_dir, f"sample_{original_idx}_variate_{v}.png")
                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"图像文件不存在: {img_path}")
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)  # (3, 224, 224)
                    sample_images.append(img_tensor)
                
                # 堆叠为 (n_variates, 3, 224, 224)
                self.cached_images.append(torch.stack(sample_images))
            
            print(f"✓ 成功从磁盘加载 {len(self.cached_images)} 个样本的图像缓存")
            return True
        except Exception as e:
            print(f"加载磁盘缓存失败: {e}")
            self.cached_images = None
            return False
    
    def _save_sample_to_disk(self, sample_idx, images):
        """保存单个样本的图像到磁盘"""
        cache_dir = self._get_cache_dir()
        to_pil = transforms.ToPILImage()
        
        for v in range(self.n_variates):
            img_tensor = images[v]  # (3, 224, 224)
            img_pil = to_pil(img_tensor)
            img_path = os.path.join(cache_dir, f"sample_{sample_idx}_variate_{v}.png")
            img_pil.save(img_path, optimize=True)
    
    def _check_disk_cache_exists(self):
        """检查磁盘缓存是否存在且完整"""
        if self.disk_cache_dir is None:
            return False
        
        cache_dir = self._get_cache_dir()
        if not os.path.exists(cache_dir):
            return False
        
        # 检查是否是扁平结构
        is_flat_structure = (cache_dir == self.disk_cache_dir)
        
        if is_flat_structure:
            # 扁平结构：检查所需的样本图像是否存在
            # 只检查几个关键样本
            sample_indices = self.indices if self.indices is not None else range(min(10, len(self.data)))
            for idx in list(sample_indices)[:5]:  # 检查前5个样本
                for v in range(self.n_variates):
                    img_path = os.path.join(cache_dir, f"sample_{idx}_variate_{v}.png")
                    if not os.path.exists(img_path):
                        return False
            return True
        else:
            # 哈希子目录结构：检查文件数
            expected_files = len(self.data) * self.n_variates
            existing_files = len([f for f in os.listdir(cache_dir) if f.endswith('.png')])
            return existing_files == expected_files
    
    def _generate_image_cache(self):
        """预生成所有图像并缓存（优先从磁盘加载）"""
        cache_dir = self._get_cache_dir()
        is_flat_structure = (cache_dir == self.disk_cache_dir)
        
        # 如果不需要加载到内存，只检查磁盘缓存是否存在
        if not self.load_to_memory:
            if self._check_disk_cache_exists():
                if is_flat_structure:
                    print(f"✓ 检测到扁平结构图像缓存，训练时将动态读取")
                else:
                    print(f"✓ 检测到磁盘缓存，训练时将动态读取")
                self.cached_images = None  # 不加载到内存
                return
            else:
                raise FileNotFoundError(
                    f"磁盘缓存不存在或不完整。\n"
                    f"预期路径: {cache_dir}\n"
                    f"请先运行 prepare_dataset.py 生成图像缓存。"
                )
        
        # 需要加载到内存，尝试从磁盘加载
        if self._load_from_disk_cache():
            return
        
        # 磁盘没有缓存，生成提示
        raise FileNotFoundError(
            f"磁盘缓存不存在。\n"
            f"预期路径: {cache_dir}\n"
            f"请先运行 prepare_dataset.py 生成图像缓存。"
        )
    
    def __len__(self):
        return len(self.data)
    
    def _load_sample_from_disk(self, idx):
        """从磁盘动态加载单个样本的图像"""
        cache_dir = self._get_cache_dir()
        transform = transforms.Compose([transforms.ToTensor()])
        
        # 如果使用了indices，需要映射到原始样本编号
        if self.indices is not None:
            original_idx = self.indices[idx]
        else:
            original_idx = idx
        
        sample_images = []
        for v in range(self.n_variates):
            img_path = os.path.join(cache_dir, f"sample_{original_idx}_variate_{v}.png")
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)  # (3, 224, 224)
            sample_images.append(img_tensor)
        
        return torch.stack(sample_images)  # (n_variates, 3, 224, 224)
    
    def truncate_time_steps(self, time_steps):
        """
        截断到指定的时间步数
        用于测试不同时间长度的性能
        
        Args:
            time_steps: 要保留的时间步数
        """
        if time_steps < self.n_time_steps:
            self.data_normalized = self.data_normalized[:, :time_steps, :]
            self.n_time_steps = time_steps
            print(f"数据已截断到 {time_steps} 步 ({time_steps*10}天)")
    
    def __getitem__(self, idx):
        """
        返回:
            x: (time_steps, n_variates) - 原始时间序列
            y: 标签
            images: (n_variates, 3, 224, 224) - 预生成的图像（如果启用缓存）
        """
        x = torch.from_numpy(self.data_normalized[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.use_cache:
            if self.cached_images is not None:
                # 从内存缓存读取
                images = self.cached_images[idx]
            else:
                # 从磁盘动态读取
                images = self._load_sample_from_disk(idx)
            return x, y, images
        else:
            return x, y


def create_classification_dataloaders_cached(
    csv_path,
    batch_size=32,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    num_workers=0,
    use_cache=True,
    disk_cache_dir="../../data/image_cache",
    load_to_memory=False
):
    """
    创建训练、验证、测试数据加载器（分层采样，图像预缓存版本）
    
    Args:
        csv_path: CSV文件路径
        batch_size: 批次大小
        test_size: 测试集比例
        val_size: 从训练集中划分的验证集比例
        random_state: 随机种子
        num_workers: 数据加载线程数
        use_cache: 是否使用图像缓存
        disk_cache_dir: 磁盘缓存目录（None则只使用内存缓存）
        load_to_memory: 是否将所有图像加载到内存（False则动态从磁盘读取，省内存）
        
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
    print("\n[1/3] 创建训练集...")
    train_dataset = CachedImageClassificationDataset(
        csv_path, indices=train_indices, use_cache=use_cache, 
        disk_cache_dir=disk_cache_dir, load_to_memory=load_to_memory
    )
    
    print("\n[2/3] 创建验证集...")
    val_dataset = CachedImageClassificationDataset(
        csv_path, indices=val_indices, use_cache=use_cache, 
        disk_cache_dir=disk_cache_dir, load_to_memory=load_to_memory
    )
    
    print("\n[3/3] 创建测试集...")
    test_dataset = CachedImageClassificationDataset(
        csv_path, indices=test_indices, use_cache=use_cache, 
        disk_cache_dir=disk_cache_dir, load_to_memory=load_to_memory
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
    if use_cache and disk_cache_dir:
        # 检查是否使用扁平结构
        test_file = os.path.join(disk_cache_dir, "sample_0_variate_0.png")
        is_flat = os.path.exists(test_file)
        
        if load_to_memory:
            print(f"加载模式: 内存缓存 (已加载到内存)")
        else:
            print(f"加载模式: 磁盘缓存 (动态读取，省内存)")
        
        print(f"缓存结构: {'扁平结构 (所有图像在同一目录)' if is_flat else '哈希子目录'}")
        print(f"缓存路径: {disk_cache_dir}")
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

