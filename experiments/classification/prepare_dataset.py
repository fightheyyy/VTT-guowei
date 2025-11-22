"""
数据集预处理脚本
预先生成所有图像并保存到磁盘，供训练时使用
"""

import pandas as pd
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
import argparse
from PIL import Image
import torchvision.transforms as transforms
import hashlib

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.preprocessor import VisualPreprocessor
from sklearn.model_selection import train_test_split


def get_cache_dir(csv_path, indices, base_dir):
    """生成与data_loader相同的缓存目录名"""
    cache_key = f"{os.path.abspath(csv_path)}_{str(sorted(indices) if indices is not None else 'all')}"
    hash_value = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    return os.path.join(base_dir, f"cache_{hash_value}")


def prepare_images(csv_path, output_dir, indices=None, n_variates=14, n_time_steps=37):
    """
    预生成所有图像并保存到磁盘
    
    Args:
        csv_path: CSV数据文件路径
        output_dir: 输出目录
        indices: 使用的样本索引（None表示全部）
        n_variates: 变量数量
        n_time_steps: 时间步数
    """
    print("="*70)
    print("图像数据集预处理")
    print("="*70)
    print(f"数据文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"时间步数: {n_time_steps}")
    print(f"变量数: {n_variates}")
    if indices is not None:
        print(f"样本索引: {len(indices)} 个")
    print("="*70)
    
    # 读取数据
    print("\n加载CSV数据...")
    df = pd.read_csv(csv_path)
    
    # 如果指定了indices，则只取这些样本
    if indices is not None:
        df = df.iloc[indices].reset_index(drop=True)
    
    labels = df.iloc[:, -1].values
    features = df.iloc[:, :-1].values
    
    # 重塑为 (n_samples, n_time_steps, n_variates)
    data = features.reshape(-1, n_time_steps, n_variates)
    
    print(f"数据集: {len(data)} 样本")
    print(f"类别分布: {np.bincount(labels)}")
    
    # 标准化
    print("\n标准化数据...")
    data_normalized = np.zeros_like(data, dtype=np.float32)
    for i in range(len(data)):
        mean = data[i].mean()
        std = data[i].std()
        if std > 0:
            data_normalized[i] = (data[i] - mean) / std
        else:
            data_normalized[i] = data[i] - mean
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化图像生成器
    print("\n初始化图像生成器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    visual_preprocessor = VisualPreprocessor(image_size=224).to(device)
    
    # 图像转换工具
    to_pil = transforms.ToPILImage()
    
    # 生成并保存图像
    print(f"\n开始生成图像（共 {len(data)} 个样本 × {n_variates} 个变量 = {len(data) * n_variates} 张图像）...")
    
    for i in tqdm(range(len(data)), desc="生成图像"):
        x = torch.from_numpy(data_normalized[i]).float().unsqueeze(0)  # (1, T, V)
        x = x.to(device)
        
        with torch.no_grad():
            images = visual_preprocessor(x)  # (1, V, 3, 224, 224)
        
        images_cpu = images.squeeze(0).cpu()  # (V, 3, 224, 224)
        
        # 保存每个变量的图像
        for v in range(n_variates):
            img_tensor = images_cpu[v]  # (3, 224, 224)
            img_pil = to_pil(img_tensor)
            img_path = os.path.join(output_dir, f"sample_{i}_variate_{v}.png")
            img_pil.save(img_path, optimize=True)
    
    # 保存元数据
    print("\n保存元数据...")
    metadata = {
        'num_samples': len(data),
        'num_variates': n_variates,
        'time_steps': n_time_steps,
        'num_classes': len(np.unique(labels)),
        'class_distribution': np.bincount(labels).tolist(),
        'labels': labels.tolist()
    }
    
    import json
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 计算总大小
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith('.png')
    )
    size_mb = total_size / (1024 * 1024)
    num_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    print("\n" + "="*70)
    print("图像生成完成！")
    print("="*70)
    print(f"输出目录: {output_dir}")
    print(f"图像数量: {num_files} 张")
    print(f"总大小: {size_mb:.1f} MB")
    print(f"元数据文件: {metadata_path}")
    print("="*70)
    
    # 清理
    del visual_preprocessor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_split_datasets(csv_path, output_base_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    生成训练集、验证集、测试集的图像（分层采样）
    
    Args:
        csv_path: CSV数据文件路径
        output_base_dir: 输出基础目录
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
    """
    print("="*70)
    print("分层数据集预处理")
    print("="*70)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, -1].values
    indices = np.arange(len(labels))
    
    # 分层划分
    print("\n划分数据集...")
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    train_labels = labels[train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=random_state, stratify=train_labels
    )
    
    print(f"训练集: {len(train_indices)} 样本")
    print(f"验证集: {len(val_indices)} 样本")
    print(f"测试集: {len(test_indices)} 样本")
    
    # 生成各个分割的图像
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    for split_name, split_indices in splits.items():
        print(f"\n处理 {split_name} 集...")
        
        # 生成与data_loader相同的哈希目录名
        output_dir = get_cache_dir(csv_path, split_indices, output_base_dir)
        
        print(f"目标目录: {output_dir}")
        
        # 生成图像
        prepare_images(csv_path, output_dir, indices=split_indices)
    
    print("\n" + "="*70)
    print("所有数据集处理完成！")
    print("="*70)
    print(f"基础目录: {output_base_dir}")
    print("训练时会自动找到对应的缓存目录")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预生成图像数据集')
    parser.add_argument('--csv_path', type=str, default='../../data/2018four.csv', help='CSV数据文件路径')
    parser.add_argument('--output_dir', type=str, default='../../data/image_cache', help='输出目录')
    parser.add_argument('--split', action='store_true', help='生成训练/验证/测试集的分层数据')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    
    args = parser.parse_args()
    
    if args.split:
        # 生成分层数据集
        prepare_split_datasets(
            args.csv_path,
            args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size
        )
    else:
        # 生成完整数据集
        prepare_images(args.csv_path, args.output_dir)

