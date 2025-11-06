"""
可变长度数据加载器
支持训练时随机采样不同的输入长度
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VariableLengthTimeseriesDataset(Dataset):
    """
    可变长度时间序列数据集
    每次随机选择输入长度，训练模型适应不同长度
    """
    
    def __init__(
        self,
        csv_paths,
        selected_bands,
        max_time_steps=36,
        min_input_length=3,
        max_input_length=30,
        fixed_input_length=None,
        normalize=True
    ):
        """
        参数:
            - csv_paths: CSV文件路径列表
            - selected_bands: 选择的波段列表
            - max_time_steps: 最大时间步数
            - min_input_length: 最小输入长度
            - max_input_length: 最大输入长度
            - fixed_input_length: 固定输入长度（测试时使用）
            - normalize: 是否归一化
        """
        self.selected_bands = selected_bands
        self.max_time_steps = max_time_steps
        self.min_input_length = min_input_length
        self.max_input_length = max_input_length
        self.fixed_input_length = fixed_input_length
        self.normalize = normalize
        
        # 加载数据
        self.data = self._load_data(csv_paths)
        
        print(f"数据集大小: {len(self.data)} 样本")
        print(f"输入长度范围: {min_input_length}-{max_input_length} 个月")
        if fixed_input_length:
            print(f"固定输入长度: {fixed_input_length} 个月")
    
    def _load_data(self, csv_paths):
        """加载CSV数据"""
        all_data = []
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            
            # 提取时间序列数据
            for band in self.selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(self.max_time_steps)]
                if not all(col in df.columns for col in cols):
                    raise ValueError(f"缺少波段 {band} 的列")
            
            # 转换为数组 [N_samples, Time_Steps, N_Variates]
            data_list = []
            for band in self.selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(self.max_time_steps)]
                band_data = df[cols].values  # [N_samples, Time_Steps]
                data_list.append(band_data)
            
            # 堆叠: [N_samples, Time_Steps, N_Variates]
            data = np.stack(data_list, axis=2)
            
            all_data.append(data)
        
        # 合并所有数据
        all_data = np.concatenate(all_data, axis=0)
        
        # 归一化
        if self.normalize:
            self.mean = np.mean(all_data, axis=(0, 1), keepdims=True)
            self.std = np.std(all_data, axis=(0, 1), keepdims=True)
            self.std = np.where(self.std < 1e-6, 1.0, self.std)
            all_data = (all_data - self.mean) / self.std
            
            print(f"数据归一化: mean={self.mean.flatten()[:3]}, std={self.std.flatten()[:3]}")
        
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        返回:
            - x: [Input_Length, N_Variates] 输入序列
            - y: [Prediction_Length, N_Variates] 目标序列
            - input_length: 输入长度（标量）
        """
        # 获取完整序列
        full_seq = self.data[idx]  # [Time_Steps, N_Variates]
        
        # 随机或固定选择输入长度
        if self.fixed_input_length is not None:
            input_length = self.fixed_input_length
        else:
            input_length = np.random.randint(
                self.min_input_length,
                self.max_input_length + 1
            )
        
        # 分割输入和目标
        x = full_seq[:input_length]  # [Input_Length, N_Variates]
        y = full_seq[input_length:]  # [Prediction_Length, N_Variates]
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.tensor(input_length, dtype=torch.long)
        )


def collate_variable_length(batch):
    """
    自定义collate函数，处理不同长度的输入
    
    由于输入长度不同，需要padding
    """
    xs, ys, input_lengths = zip(*batch)
    
    # 找到最大长度
    max_input_len = max(x.shape[0] for x in xs)
    max_pred_len = max(y.shape[0] for y in ys)
    n_variates = xs[0].shape[1]
    
    batch_size = len(xs)
    
    # Padding输入
    x_padded = torch.zeros(batch_size, max_input_len, n_variates)
    for i, x in enumerate(xs):
        x_padded[i, :x.shape[0], :] = x
    
    # Padding目标
    y_padded = torch.zeros(batch_size, max_pred_len, n_variates)
    for i, y in enumerate(ys):
        y_padded[i, :y.shape[0], :] = y
    
    input_lengths = torch.stack(input_lengths)
    
    return x_padded, y_padded, input_lengths


def create_variable_length_dataloaders(
    train_csv_paths,
    test_csv_paths,
    selected_bands,
    max_time_steps=36,
    min_input_length=3,
    max_input_length=30,
    test_input_length=18,
    batch_size=16,
    num_workers=0
):
    """
    创建可变长度数据加载器
    
    参数:
        - train_csv_paths: 训练集CSV路径列表
        - test_csv_paths: 测试集CSV路径列表
        - selected_bands: 选择的波段
        - max_time_steps: 最大时间步数
        - min_input_length: 训练时最小输入长度
        - max_input_length: 训练时最大输入长度
        - test_input_length: 测试时固定输入长度
        - batch_size: batch大小
    
    返回:
        - train_loader: 训练数据加载器
        - test_loader: 测试数据加载器
        - n_variates: 变量数量
    """
    # 训练集：随机输入长度
    train_dataset = VariableLengthTimeseriesDataset(
        csv_paths=train_csv_paths,
        selected_bands=selected_bands,
        max_time_steps=max_time_steps,
        min_input_length=min_input_length,
        max_input_length=max_input_length,
        fixed_input_length=None,  # 训练时随机
        normalize=True
    )
    
    # 测试集：固定输入长度
    test_dataset = VariableLengthTimeseriesDataset(
        csv_paths=test_csv_paths,
        selected_bands=selected_bands,
        max_time_steps=max_time_steps,
        min_input_length=min_input_length,
        max_input_length=max_input_length,
        fixed_input_length=test_input_length,  # 测试时固定
        normalize=True
    )
    
    # 使用训练集的归一化参数
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_variable_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_length
    )
    
    n_variates = len(selected_bands)
    
    return train_loader, test_loader, n_variates


if __name__ == "__main__":
    # 测试数据加载器
    print("=" * 70)
    print("测试可变长度数据加载器")
    print("=" * 70)
    
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = [
        "extract2022_20251010_165007.csv"
    ]
    
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    
    train_loader, test_loader, n_variates = create_variable_length_dataloaders(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        max_time_steps=36,
        min_input_length=3,
        max_input_length=30,
        test_input_length=18,
        batch_size=4
    )
    
    print(f"\n变量数量: {n_variates}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试一个batch
    print("\n" + "=" * 70)
    print("测试训练批次（随机输入长度）")
    print("=" * 70)
    
    for i, (x, y, input_lengths) in enumerate(train_loader):
        print(f"\nBatch {i+1}:")
        print(f"  输入形状: {x.shape}")
        print(f"  目标形状: {y.shape}")
        print(f"  输入长度: {input_lengths.tolist()}")
        print(f"  预测长度: {[36 - l.item() for l in input_lengths]}")
        
        if i >= 2:
            break
    
    # 测试测试批次
    print("\n" + "=" * 70)
    print("测试测试批次（固定输入长度=18）")
    print("=" * 70)
    
    for i, (x, y, input_lengths) in enumerate(test_loader):
        print(f"\nBatch {i+1}:")
        print(f"  输入形状: {x.shape}")
        print(f"  目标形状: {y.shape}")
        print(f"  输入长度: {input_lengths[0].item()}")
        print(f"  预测长度: {36 - input_lengths[0].item()}")
        
        if i >= 1:
            break

