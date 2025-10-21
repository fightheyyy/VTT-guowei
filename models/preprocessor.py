"""
数据预处理器模块
将输入的数值时间序列转换为视觉和语言两种模态
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
from torchvision import transforms


class VisualPreprocessor(nn.Module):
    """
    将时间序列可视化为图像
    输入: [Batch, Time_Steps, N_Variates]
    输出: [Batch, N_Variates, 3, 224, 224]
    """
    
    def __init__(self, image_size=224, colors=None):
        super().__init__()
        self.image_size = image_size
        
        # 预定义高对比度颜色列表
        if colors is None:
            self.colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
            ]
        else:
            self.colors = colors
        
        # 图像转换pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def instance_normalize(self, x):
        """
        实例归一化：在Time_Steps维度上进行归一化
        输入: [Batch, Time_Steps, N_Variates]
        输出: [Batch, Time_Steps, N_Variates]
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std
    
    def plot_to_tensor(self, data, color):
        """
        将单个变量的时间序列绘制为图像张量
        输入: data [Time_Steps], color: str
        输出: tensor [3, H, W]
        """
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        
        # 绘制纯净的折线图
        ax.plot(data, color=color, linewidth=2)
        
        # 移除所有坐标轴、网格、边框
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # 紧凑布局
        plt.tight_layout(pad=0)
        
        # 保存到buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # 转换为PIL Image
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        
        # 转换为tensor
        img_tensor = self.transform(img)
        buf.close()
        
        return img_tensor
    
    def forward(self, x):
        """
        前向传播
        输入: x [Batch, Time_Steps, N_Variates]
        输出: images [Batch, N_Variates, 3, H, W]
        """
        batch_size, time_steps, n_variates = x.shape
        
        # 实例归一化
        x_norm = self.instance_normalize(x)  # [Batch, Time_Steps, N_Variates]
        
        # 将数据转移到CPU并转换为numpy（matplotlib需要）
        x_np = x_norm.detach().cpu().numpy()
        
        # 为每个样本的每个变量生成图像
        images = []
        for b in range(batch_size):
            batch_images = []
            for v in range(n_variates):
                color = self.colors[v % len(self.colors)]
                img_tensor = self.plot_to_tensor(x_np[b, :, v], color)
                batch_images.append(img_tensor)
            images.append(torch.stack(batch_images))  # [N_Variates, 3, H, W]
        
        images = torch.stack(images)  # [Batch, N_Variates, 3, H, W]
        
        return images.to(x.device)


class LanguagePreprocessor(nn.Module):
    """
    将时间序列分块处理用于语言模块
    输入: [Batch, Time_Steps, N_Variates]
    输出: [Batch, N_Variates, N_Patches, Patch_Length]
    """
    
    def __init__(self, patch_length=16, stride=8):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
    
    def instance_normalize(self, x):
        """
        实例归一化：在Time_Steps维度上进行归一化
        输入: [Batch, Time_Steps, N_Variates]
        输出: [Batch, Time_Steps, N_Variates]
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std
    
    def forward(self, x):
        """
        前向传播
        输入: x [Batch, Time_Steps, N_Variates]
        输出: patches [Batch, N_Variates, N_Patches, Patch_Length]
        """
        batch_size, time_steps, n_variates = x.shape
        
        # 实例归一化
        x_norm = self.instance_normalize(x)  # [Batch, Time_Steps, N_Variates]
        
        # 转置为 [Batch, N_Variates, Time_Steps] 便于处理
        x_norm = x_norm.transpose(1, 2)
        
        # 使用unfold进行分块
        # unfold(dimension, size, step)
        patches = x_norm.unfold(dimension=2, size=self.patch_length, step=self.stride)
        # 输出: [Batch, N_Variates, N_Patches, Patch_Length]
        
        return patches

