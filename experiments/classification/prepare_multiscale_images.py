"""
预生成多时间尺度图像
为不同时间长度的序列预先生成图像，加速dual_modal训练
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.preprocessor import VisualPreprocessor


def prepare_multiscale_images(
    csv_path,
    output_dir,
    time_steps_list=[3, 6, 9, 12, 15, 18],  # 默认6个时间尺度（减少50%）
    n_variates=14
):
    """
    预生成多时间尺度的图像
    
    Args:
        csv_path: CSV数据路径
        output_dir: 输出目录
        time_steps_list: 要生成的时间长度列表（默认6个，覆盖早期到中期）
        n_variates: 变量数
    """
    print("="*70)
    print("多时间尺度图像预生成 - Dual Modal训练加速")
    print("="*70)
    print(f"数据文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"时间尺度: {time_steps_list}")
    print(f"变量数: {n_variates}")
    print("="*70)
    
    # 读取数据
    print("\n[1/4] 加载数据...")
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, -1].values
    features = df.iloc[:, :-1].values
    
    # 重塑为 (n_samples, 37, n_variates)
    full_data = features.reshape(-1, 37, n_variates)
    
    print(f"  样本数: {len(full_data)}")
    print(f"  类别数: {len(np.unique(labels))}")
    print(f"  类别分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 标准化
    print("\n[2/4] 标准化数据...")
    data_normalized = np.zeros_like(full_data, dtype=np.float32)
    for i in range(len(full_data)):
        mean = full_data[i].mean()
        std = full_data[i].std()
        if std > 0:
            data_normalized[i] = (full_data[i] - mean) / std
        else:
            data_normalized[i] = full_data[i] - mean
    
    # 初始化图像生成器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3/4] 初始化图像生成器...")
    print(f"  设备: {device}")
    visual_preprocessor = VisualPreprocessor(image_size=224).to(device)
    to_pil = transforms.ToPILImage()
    
    # 创建输出目录结构
    for time_steps in time_steps_list:
        time_dir = os.path.join(output_dir, f"time_{time_steps}")
        os.makedirs(time_dir, exist_ok=True)
    
    # 计算总量
    total_images = len(full_data) * len(time_steps_list) * n_variates
    print(f"\n[4/4] 开始生成图像...")
    print(f"  样本数: {len(full_data)}")
    print(f"  时间尺度: {len(time_steps_list)} 个")
    print(f"  变量数: {n_variates}")
    print(f"  总图像数: {total_images:,} 张")
    print(f"  预计磁盘: {total_images * 50 / 1024:.1f} MB")
    print()
    
    # 生成图像
    with tqdm(total=len(full_data) * len(time_steps_list), 
              desc="生成进度", 
              unit="样本×时间尺度") as pbar:
        for sample_idx in range(len(full_data)):
            # 为每个时间长度生成图像
            for time_steps in time_steps_list:
                # 截断到指定时间步
                truncated_data = data_normalized[sample_idx, :time_steps, :]
                
                # 转为tensor
                x = torch.from_numpy(truncated_data).float().unsqueeze(0)  # [1, T, V]
                x = x.to(device)
                
                # 生成图像
                with torch.no_grad():
                    images = visual_preprocessor(x)  # [1, V, 3, 224, 224]
                
                images_cpu = images.squeeze(0).cpu()  # [V, 3, 224, 224]
                
                # 保存每个变量的图像
                time_dir = os.path.join(output_dir, f"time_{time_steps}")
                for v in range(n_variates):
                    img_tensor = images_cpu[v]  # [3, 224, 224]
                    img_pil = to_pil(img_tensor)
                    img_path = os.path.join(time_dir, f"sample_{sample_idx}_variate_{v}.png")
                    img_pil.save(img_path, optimize=True)
                
                pbar.update(1)
    
    # 保存元数据
    print("\n保存元数据...")
    import json
    metadata = {
        'num_samples': len(full_data),
        'time_steps_list': time_steps_list,
        'n_variates': n_variates,
        'num_classes': len(np.unique(labels)),
        'class_distribution': np.bincount(labels).tolist(),
        'total_images': total_images,
        'note': '多时间尺度缓存用于dual_modal训练的时间masking增强'
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 统计各目录大小
    print("\n各时间尺度缓存大小:")
    total_size = 0
    for time_steps in time_steps_list:
        time_dir = os.path.join(output_dir, f"time_{time_steps}")
        dir_size = sum(
            os.path.getsize(os.path.join(time_dir, f))
            for f in os.listdir(time_dir) if f.endswith('.png')
        )
        total_size += dir_size
        print(f"  time_{time_steps:2d}: {dir_size / (1024*1024):6.1f} MB ({len(full_data) * n_variates:,} 张)")
    
    print(f"\n总大小: {total_size / (1024*1024):.1f} MB = {total_size / (1024**3):.2f} GB")
    print("\n" + "="*70)
    print("图像生成完成！")
    print("="*70)
    print(f"输出目录: {output_dir}")
    print(f"子目录: {', '.join([f'time_{t}' for t in time_steps_list])}")
    print(f"总图像数: {total_images:,} 张")
    print("="*70)
    
    # 清理
    del visual_preprocessor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MultiScaleImageCache:
    """
    多时间尺度图像缓存加载器
    训练时根据时间长度选择对应的预生成图像
    """
    def __init__(self, cache_dir, time_steps_list, n_variates=14):
        self.cache_dir = cache_dir
        self.time_steps_list = sorted(time_steps_list)
        self.n_variates = n_variates
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # 验证缓存存在
        for time_steps in time_steps_list:
            time_dir = os.path.join(cache_dir, f"time_{time_steps}")
            if not os.path.exists(time_dir):
                raise FileNotFoundError(f"缓存目录不存在: {time_dir}")
        
        print(f"MultiScaleImageCache初始化完成")
        print(f"  缓存目录: {cache_dir}")
        print(f"  支持的时间尺度: {time_steps_list}")
    
    def get_closest_time_steps(self, target_steps):
        """找到最接近的预生成时间长度（小于等于target_steps）"""
        # 找到小于等于target_steps的最大值
        valid_steps = [t for t in self.time_steps_list if t <= target_steps]
        if valid_steps:
            return max(valid_steps)
        else:
            # 如果target_steps比最小的还小，返回最小的
            return self.time_steps_list[0]
    
    def load_images(self, sample_idx, time_steps):
        """
        加载指定样本和时间长度的图像
        
        Args:
            sample_idx: 样本索引
            time_steps: 目标时间步数
        
        Returns:
            images: [V, 3, 224, 224] tensor
        """
        # 找到最接近的预生成长度
        cached_steps = self.get_closest_time_steps(time_steps)
        time_dir = os.path.join(self.cache_dir, f"time_{cached_steps}")
        
        images = []
        for v in range(self.n_variates):
            img_path = os.path.join(time_dir, f"sample_{sample_idx}_variate_{v}.png")
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)  # [3, 224, 224]
            images.append(img_tensor)
        
        return torch.stack(images)  # [V, 3, 224, 224]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预生成多时间尺度图像')
    parser.add_argument('--csv_path', type=str, default='../../data/2018four.csv', 
                       help='CSV数据路径')
    parser.add_argument('--output_dir', type=str, default='../../data/multiscale_image_cache',
                       help='输出目录')
    parser.add_argument('--full', action='store_true',
                       help='完整模式：生成12个时间尺度（3,6,9,12,15,18,21,24,27,30,33,37）')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式：只生成3个时间尺度（6,12,18）用于测试')
    
    args = parser.parse_args()
    
    # 根据模式选择时间尺度
    if args.quick:
        time_steps_list = [6, 12, 18]
        print("快速测试模式：只生成3个时间尺度")
    elif args.full:
        time_steps_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 37]
        print("完整模式：生成12个时间尺度")
    else:
        # 默认模式：6个时间尺度，覆盖早期识别的关键时期
        time_steps_list = [3, 6, 9, 12, 15, 18]
        print("默认模式：生成6个时间尺度（早期+中期）")
    
    print(f"选择的时间尺度: {time_steps_list}")
    print()
    
    prepare_multiscale_images(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        time_steps_list=time_steps_list
    )
    
    # 测试加载
    print("\n测试缓存加载...")
    try:
        cache = MultiScaleImageCache(args.output_dir, time_steps_list)
        
        # 测试加载不同时间长度
        test_cases = [
            (0, 3),   # 最短
            (0, 10),  # 中间（会自动选择最接近的9）
            (0, 18),  # 最长
        ]
        
        for sample_idx, time_steps in test_cases:
            images = cache.load_images(sample_idx, time_steps)
            closest = cache.get_closest_time_steps(time_steps)
            print(f"  ✓ 样本{sample_idx}, 目标{time_steps}步 → 加载time_{closest}: {images.shape}")
        
        print("\n缓存加载测试通过！")
        
    except Exception as e:
        print(f"\n✗ 缓存加载测试失败: {e}")

