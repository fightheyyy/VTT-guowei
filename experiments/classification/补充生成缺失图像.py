"""
补充生成缺失的多时间尺度图像
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.preprocessor import VisualPreprocessor


def find_missing_samples(output_dir, total_samples, time_steps_list):
    """找出缺失的样本编号"""
    missing = set()
    
    # 检查每个时间尺度目录
    for time_steps in time_steps_list:
        time_dir = os.path.join(output_dir, f"time_{time_steps}")
        if not os.path.exists(time_dir):
            print(f"警告: 目录不存在 {time_dir}")
            continue
            
        # 检查每个样本
        for sample_idx in range(total_samples):
            # 检查第一个变量的图像是否存在
            img_path = os.path.join(time_dir, f"sample_{sample_idx}_variate_0.png")
            if not os.path.exists(img_path):
                missing.add(sample_idx)
    
    return sorted(list(missing))


def generate_missing_images(
    csv_path,
    output_dir,
    missing_indices,
    time_steps_list=[3, 6, 9, 12, 15, 18],
    n_variates=14
):
    """
    为缺失的样本生成图像
    """
    print("="*70)
    print("补充生成缺失图像")
    print("="*70)
    print(f"数据文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"缺失样本: {len(missing_indices)} 个")
    print(f"范围: {missing_indices[0]} ~ {missing_indices[-1]}")
    print("="*70)
    
    # 读取数据
    print("\n[1/4] 加载数据...")
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, -1].values
    features = df.iloc[:, :-1].values
    
    # 重塑为 (n_samples, 37, n_variates)
    full_data = features.reshape(-1, 37, n_variates)
    
    print(f"  总样本数: {len(full_data)}")
    print(f"  需要生成的样本数: {len(missing_indices)}")
    
    # 标准化
    print("\n[2/4] 标准化缺失样本数据...")
    data_normalized = np.zeros((len(missing_indices), 37, n_variates), dtype=np.float32)
    for i, sample_idx in enumerate(missing_indices):
        mean = full_data[sample_idx].mean()
        std = full_data[sample_idx].std()
        if std > 0:
            data_normalized[i] = (full_data[sample_idx] - mean) / std
        else:
            data_normalized[i] = full_data[sample_idx] - mean
    
    # 初始化图像生成器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3/4] 初始化图像生成器...")
    print(f"  设备: {device}")
    visual_preprocessor = VisualPreprocessor(image_size=224).to(device)
    to_pil = transforms.ToPILImage()
    
    # 确保输出目录存在
    for time_steps in time_steps_list:
        time_dir = os.path.join(output_dir, f"time_{time_steps}")
        os.makedirs(time_dir, exist_ok=True)
    
    # 计算总量
    total_images = len(missing_indices) * len(time_steps_list) * n_variates
    print(f"\n[4/4] 开始生成缺失图像...")
    print(f"  样本数: {len(missing_indices)}")
    print(f"  时间尺度: {len(time_steps_list)} 个")
    print(f"  变量数: {n_variates}")
    print(f"  总图像数: {total_images:,} 张")
    print()
    
    # 生成图像
    with tqdm(total=len(missing_indices) * len(time_steps_list), 
              desc="生成进度", 
              unit="样本×时间尺度") as pbar:
        for i, sample_idx in enumerate(missing_indices):
            # 为每个时间长度生成图像
            for time_steps in time_steps_list:
                # 截断到指定时间步
                truncated_data = data_normalized[i, :time_steps, :]
                
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
    
    print("\n" + "="*70)
    print("缺失图像补充完成！")
    print("="*70)
    print(f"补充样本数: {len(missing_indices)}")
    print(f"补充图像数: {total_images:,} 张")
    print("="*70)
    
    # 清理
    del visual_preprocessor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    csv_path = '../../data/2018four.csv'
    output_dir = '../../data/multiscale_image_cache'
    time_steps_list = [3, 6, 9, 12, 15, 18]
    
    # 读取CSV获取总样本数
    print("检查数据文件...")
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"CSV文件总样本数: {total_samples}")
    
    # 查找缺失的样本
    print("\n检查缺失样本...")
    missing_indices = find_missing_samples(output_dir, total_samples, time_steps_list)
    
    if len(missing_indices) == 0:
        print("没有缺失样本，所有图像已完整生成！")
    else:
        print(f"发现 {len(missing_indices)} 个样本的图像缺失")
        print(f"缺失范围: sample_{missing_indices[0]} ~ sample_{missing_indices[-1]}")
        
        # 补充生成
        generate_missing_images(
            csv_path=csv_path,
            output_dir=output_dir,
            missing_indices=missing_indices,
            time_steps_list=time_steps_list
        )
        
        # 验证
        print("\n验证补充结果...")
        still_missing = find_missing_samples(output_dir, total_samples, time_steps_list)
        if len(still_missing) == 0:
            print("验证通过！所有图像已完整！")
        else:
            print(f"警告: 仍有 {len(still_missing)} 个样本缺失")

