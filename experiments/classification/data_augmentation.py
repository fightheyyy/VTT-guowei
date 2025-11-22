"""
时间序列数据增强模块
针对2018four.csv数据集设计的增强策略
"""

import torch
import numpy as np


class TimeSeriesAugmentation:
    """
    时间序列数据增强类
    适用于遥感时序数据（14变量 × 12时间步）
    """
    
    def __init__(self, 
                 noise_std=0.02,
                 scale_range=(0.9, 1.1),
                 shift_max=2,
                 variate_dropout_prob=0.1):
        """
        参数:
            noise_std: 高斯噪声标准差
            scale_range: 随机缩放范围 (min, max)
            shift_max: 时间平移最大步数
            variate_dropout_prob: 变量dropout概率
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.variate_dropout_prob = variate_dropout_prob
    
    def add_gaussian_noise(self, x):
        """
        添加高斯噪声
        Args:
            x: [Batch, Time, Variates] 或 [Time, Variates]
        Returns:
            x_augmented: 添加噪声后的数据
        """
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def random_scaling(self, x):
        """
        随机缩放（按变量独立缩放）
        Args:
            x: [Batch, Time, Variates]
        Returns:
            x_scaled: 缩放后的数据
        """
        if x.dim() == 2:  # [Time, Variates]
            x = x.unsqueeze(0)
            squeeze_needed = True
        else:
            squeeze_needed = False
        
        batch_size, time_steps, n_variates = x.shape
        
        # 为每个样本的每个变量生成独立的缩放因子
        scale_factors = torch.FloatTensor(batch_size, 1, n_variates).uniform_(
            self.scale_range[0], self.scale_range[1]
        ).to(x.device)
        
        x_scaled = x * scale_factors
        
        if squeeze_needed:
            x_scaled = x_scaled.squeeze(0)
        
        return x_scaled
    
    def time_shift(self, x):
        """
        时间平移（沿时间轴循环移位）
        Args:
            x: [Batch, Time, Variates]
        Returns:
            x_shifted: 平移后的数据
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_needed = True
        else:
            squeeze_needed = False
        
        batch_size, time_steps, n_variates = x.shape
        
        # 随机平移步数
        shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
        
        if shift != 0:
            x_shifted = torch.roll(x, shifts=shift, dims=1)
        else:
            x_shifted = x.clone()
        
        if squeeze_needed:
            x_shifted = x_shifted.squeeze(0)
        
        return x_shifted
    
    def variate_dropout(self, x):
        """
        变量dropout（随机将某些变量置零）
        Args:
            x: [Batch, Time, Variates]
        Returns:
            x_dropped: dropout后的数据
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_needed = True
        else:
            squeeze_needed = False
        
        batch_size, time_steps, n_variates = x.shape
        
        # 为每个样本的每个变量生成dropout mask
        dropout_mask = torch.bernoulli(
            torch.ones(batch_size, 1, n_variates) * (1 - self.variate_dropout_prob)
        ).to(x.device)
        
        x_dropped = x * dropout_mask
        
        if squeeze_needed:
            x_dropped = x_dropped.squeeze(0)
        
        return x_dropped
    
    def temporal_cutout(self, x, max_cutout_length=3):
        """
        时间截断（随机将某段时间置零，模拟缺失数据）
        Args:
            x: [Batch, Time, Variates]
            max_cutout_length: 最大截断长度
        Returns:
            x_cutout: 截断后的数据
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_needed = True
        else:
            squeeze_needed = False
        
        batch_size, time_steps, n_variates = x.shape
        x_cutout = x.clone()
        
        # 随机选择截断起点和长度
        cutout_length = torch.randint(1, max_cutout_length + 1, (1,)).item()
        if time_steps > cutout_length:
            start_idx = torch.randint(0, time_steps - cutout_length, (1,)).item()
            x_cutout[:, start_idx:start_idx+cutout_length, :] = 0
        
        if squeeze_needed:
            x_cutout = x_cutout.squeeze(0)
        
        return x_cutout
    
    def mixup(self, x1, x2, alpha=0.2):
        """
        Mixup增强（线性混合两个样本）
        Args:
            x1, x2: [Time, Variates]
            alpha: Beta分布参数
        Returns:
            x_mixed: 混合后的样本
            lam: 混合系数
        """
        lam = np.random.beta(alpha, alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        return x_mixed, lam
    
    def __call__(self, x, augment_prob=0.7, augment_types=['noise', 'scale']):
        """
        随机应用增强
        Args:
            x: [Batch, Time, Variates] 或 [Time, Variates]
            augment_prob: 应用增强的概率
            augment_types: 增强类型列表，可选:
                - 'noise': 高斯噪声
                - 'scale': 随机缩放
                - 'shift': 时间平移
                - 'variate_drop': 变量dropout
                - 'temporal_cutout': 时间截断
        Returns:
            x_augmented: 增强后的数据
        """
        x_aug = x.clone()
        
        for aug_type in augment_types:
            if torch.rand(1).item() < augment_prob:
                if aug_type == 'noise':
                    x_aug = self.add_gaussian_noise(x_aug)
                elif aug_type == 'scale':
                    x_aug = self.random_scaling(x_aug)
                elif aug_type == 'shift':
                    x_aug = self.time_shift(x_aug)
                elif aug_type == 'variate_drop':
                    x_aug = self.variate_dropout(x_aug)
                elif aug_type == 'temporal_cutout':
                    x_aug = self.temporal_cutout(x_aug)
        
        return x_aug


class ImageAugmentation:
    """
    图像数据增强（针对预缓存的时序图像）
    由于图像已预缓存，这里提供一些轻量级的增强
    """
    
    def __init__(self, brightness_range=0.1, contrast_range=0.1):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def random_brightness(self, images):
        """
        随机亮度调整
        Args:
            images: [Batch, N_Variates, 3, H, W]
        Returns:
            images_adjusted: 调整后的图像
        """
        factor = 1 + torch.FloatTensor(1).uniform_(
            -self.brightness_range, self.brightness_range
        ).to(images.device)
        
        return torch.clamp(images * factor, 0, 1)
    
    def random_contrast(self, images):
        """
        随机对比度调整
        Args:
            images: [Batch, N_Variates, 3, H, W]
        Returns:
            images_adjusted: 调整后的图像
        """
        mean = images.mean(dim=(-2, -1), keepdim=True)
        factor = 1 + torch.FloatTensor(1).uniform_(
            -self.contrast_range, self.contrast_range
        ).to(images.device)
        
        return torch.clamp((images - mean) * factor + mean, 0, 1)
    
    def __call__(self, images, augment_prob=0.5):
        """
        应用图像增强
        Args:
            images: [Batch, N_Variates, 3, H, W]
            augment_prob: 增强概率
        Returns:
            images_augmented: 增强后的图像
        """
        images_aug = images.clone()
        
        if torch.rand(1).item() < augment_prob:
            images_aug = self.random_brightness(images_aug)
        
        if torch.rand(1).item() < augment_prob:
            images_aug = self.random_contrast(images_aug)
        
        return images_aug


# ============ 便捷函数 ============

def get_augmentation_pipeline(mode='light'):
    """
    获取预配置的增强管道
    
    Args:
        mode: 'light', 'medium', 'heavy'
    
    Returns:
        ts_aug: 时序数据增强器
        img_aug: 图像增强器
        config: 配置字典
    """
    if mode == 'light':
        ts_aug = TimeSeriesAugmentation(
            noise_std=0.01,
            scale_range=(0.95, 1.05),
            shift_max=1,
            variate_dropout_prob=0.05
        )
        img_aug = ImageAugmentation(
            brightness_range=0.05,
            contrast_range=0.05
        )
        config = {
            'ts_prob': 0.5,
            'ts_types': ['noise', 'scale'],
            'img_prob': 0.3
        }
    
    elif mode == 'medium':
        ts_aug = TimeSeriesAugmentation(
            noise_std=0.02,
            scale_range=(0.9, 1.1),
            shift_max=2,
            variate_dropout_prob=0.1
        )
        img_aug = ImageAugmentation(
            brightness_range=0.1,
            contrast_range=0.1
        )
        config = {
            'ts_prob': 0.7,
            'ts_types': ['noise', 'scale', 'shift'],
            'img_prob': 0.5
        }
    
    elif mode == 'heavy':
        ts_aug = TimeSeriesAugmentation(
            noise_std=0.03,
            scale_range=(0.85, 1.15),
            shift_max=3,
            variate_dropout_prob=0.15
        )
        img_aug = ImageAugmentation(
            brightness_range=0.15,
            contrast_range=0.15
        )
        config = {
            'ts_prob': 0.8,
            'ts_types': ['noise', 'scale', 'shift', 'variate_drop', 'temporal_cutout'],
            'img_prob': 0.7
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return ts_aug, img_aug, config


if __name__ == "__main__":
    # 测试代码
    print("测试时序数据增强...")
    x = torch.randn(32, 12, 14)  # [Batch, Time, Variates]
    
    ts_aug, img_aug, config = get_augmentation_pipeline(mode='medium')
    
    x_augmented = ts_aug(x, 
                         augment_prob=config['ts_prob'],
                         augment_types=config['ts_types'])
    
    print(f"原始数据形状: {x.shape}")
    print(f"增强数据形状: {x_augmented.shape}")
    print(f"数据变化幅度: {(x_augmented - x).abs().mean().item():.6f}")
    
    print("\n测试图像增强...")
    images = torch.rand(32, 14, 3, 224, 224)  # [Batch, N_Variates, 3, H, W]
    images_augmented = img_aug(images, augment_prob=config['img_prob'])
    
    print(f"原始图像形状: {images.shape}")
    print(f"增强图像形状: {images_augmented.shape}")
    print(f"图像变化幅度: {(images_augmented - images).abs().mean().item():.6f}")
    
    print("\n✅ 增强模块测试通过!")

