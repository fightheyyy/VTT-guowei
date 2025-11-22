"""
改进的损失函数
用于提升早期分类性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAwareFocalLoss(nn.Module):
    """
    时间感知的Focal Loss
    早期时间步的困难样本给更高权重
    """
    def __init__(self, alpha=0.25, gamma=2.0, time_weight_factor=2.0, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.time_weight_factor = time_weight_factor
        self.num_classes = num_classes
    
    def forward(self, logits, targets, time_ratio):
        """
        Args:
            logits: [B, num_classes] 预测logits
            targets: [B] 真实标签
            time_ratio: float, 当前使用的时间比例 (0-1)，越小表示越早期
        
        Returns:
            loss: scalar
        """
        # 计算交叉熵（不reduction）
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # Focal loss基础项
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 时间加权: 早期预测困难，给更高权重
        # time_ratio接近0时（早期），weight最大
        # time_ratio接近1时（后期），weight接近1
        time_weight = 1.0 + self.time_weight_factor * (1.0 - time_ratio)
        
        return (focal_loss * time_weight).mean()


class EarlyFeatureContrastiveLoss(nn.Module):
    """
    早期特征增强的对比学习损失
    在早期阶段强化视觉-语言特征对齐
    """
    def __init__(self, temperature=0.07, early_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.early_weight = early_weight
    
    def forward(self, features_visual, features_language, time_ratio):
        """
        Args:
            features_visual: [B, D] 视觉特征
            features_language: [B, D] 语言特征
            time_ratio: float, 当前时间比例 (0-1)
        
        Returns:
            loss: scalar
        """
        batch_size = features_visual.size(0)
        
        # L2归一化
        features_visual = F.normalize(features_visual, dim=-1)
        features_language = F.normalize(features_language, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features_visual, features_language.T) / self.temperature
        
        # 对比损失（对称）
        labels = torch.arange(batch_size).to(features_visual.device)
        loss_v2l = F.cross_entropy(similarity, labels)
        loss_l2v = F.cross_entropy(similarity.T, labels)
        
        contrastive_loss = (loss_v2l + loss_l2v) / 2
        
        # 早期特征给更高权重
        weight = 1.0 + self.early_weight * (1.0 - time_ratio)
        
        return contrastive_loss * weight


class CombinedEarlyLoss(nn.Module):
    """
    组合损失：分类 + 对比学习
    """
    def __init__(self, 
                 num_classes=4,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 time_weight_factor=2.0,
                 contrastive_temp=0.07,
                 contrastive_early_weight=2.0,
                 contrastive_weight=0.1):
        super().__init__()
        
        self.classification_loss = TimeAwareFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            time_weight_factor=time_weight_factor,
            num_classes=num_classes
        )
        
        self.contrastive_loss = EarlyFeatureContrastiveLoss(
            temperature=contrastive_temp,
            early_weight=contrastive_early_weight
        )
        
        self.contrastive_weight = contrastive_weight
    
    def forward(self, logits, targets, 
                features_visual=None, features_language=None, 
                time_ratio=1.0):
        """
        Args:
            logits: 分类logits
            targets: 真实标签
            features_visual: 视觉特征（可选）
            features_language: 语言特征（可选）
            time_ratio: 时间比例
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典（用于记录）
        """
        # 分类损失
        cls_loss = self.classification_loss(logits, targets, time_ratio)
        
        # 对比损失
        if features_visual is not None and features_language is not None:
            contr_loss = self.contrastive_loss(features_visual, features_language, time_ratio)
            total_loss = cls_loss + self.contrastive_weight * contr_loss
        else:
            contr_loss = torch.tensor(0.0).to(cls_loss.device)
            total_loss = cls_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'contrastive': contr_loss.item(),
            'time_ratio': time_ratio
        }
        
        return total_loss, loss_dict


def temporal_masking_augmentation(x, min_ratio=0.2, max_ratio=1.0, mode='truncate'):
    """
    时间序列增强：随机截断
    
    Args:
        x: [B, T, V] 时间序列
        min_ratio: 最少保留比例
        max_ratio: 最多保留比例
        mode: 'truncate' 或 'zero_pad'
    
    Returns:
        x_masked: [B, T, V] 增强后的序列
        keep_steps: int, 保留的时间步数
        time_ratio: float, 保留比例
    """
    batch_size, time_steps, n_variates = x.shape
    
    # 随机选择保留比例
    keep_ratio = torch.rand(1).item() * (max_ratio - min_ratio) + min_ratio
    keep_steps = max(int(time_steps * keep_ratio), 3)  # 至少保留3步
    
    if mode == 'truncate':
        # 截断模式：只保留前keep_steps
        x_masked = x[:, :keep_steps, :].clone()
        
        # 用零填充到原长度（如果需要保持维度）
        if keep_steps < time_steps:
            padding = torch.zeros(batch_size, time_steps - keep_steps, n_variates).to(x.device)
            x_masked = torch.cat([x_masked, padding], dim=1)
    
    elif mode == 'zero_pad':
        # 零填充模式：后面位置置零
        x_masked = x.clone()
        if keep_steps < time_steps:
            x_masked[:, keep_steps:, :] = 0
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return x_masked, keep_steps, keep_ratio


class CurriculumScheduler:
    """
    课程学习调度器
    控制训练过程中时间masking的范围
    """
    def __init__(self, total_epochs=100, warmup_epochs=20, 
                 min_ratio_start=0.7, min_ratio_end=0.2):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_ratio_start = min_ratio_start
        self.min_ratio_end = min_ratio_end
    
    def get_time_range(self, current_epoch):
        """
        返回当前epoch应该使用的时间步范围
        
        Args:
            current_epoch: 当前epoch
        
        Returns:
            min_ratio: 最小保留比例
            max_ratio: 最大保留比例
        """
        if current_epoch < self.warmup_epochs:
            # 热身阶段：只用长序列
            min_ratio = self.min_ratio_start
            max_ratio = 1.0
        else:
            # 逐渐引入短序列
            progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            
            # 线性衰减
            min_ratio = self.min_ratio_start - (self.min_ratio_start - self.min_ratio_end) * progress
            max_ratio = 1.0
        
        return min_ratio, max_ratio
    
    def __repr__(self):
        return (f"CurriculumScheduler(total_epochs={self.total_epochs}, "
                f"warmup_epochs={self.warmup_epochs}, "
                f"range={self.min_ratio_start:.1f}->{self.min_ratio_end:.1f})")


if __name__ == "__main__":
    # 测试损失函数
    print("测试TimeAwareFocalLoss")
    loss_fn = TimeAwareFocalLoss()
    
    logits = torch.randn(32, 4)
    targets = torch.randint(0, 4, (32,))
    
    # 早期：time_ratio=0.3
    loss_early = loss_fn(logits, targets, time_ratio=0.3)
    print(f"Early loss (time_ratio=0.3): {loss_early.item():.4f}")
    
    # 后期：time_ratio=0.9
    loss_late = loss_fn(logits, targets, time_ratio=0.9)
    print(f"Late loss (time_ratio=0.9): {loss_late.item():.4f}")
    
    print(f"\nLoss ratio (early/late): {loss_early.item()/loss_late.item():.2f}x")
    
    # 测试时间masking
    print("\n测试temporal_masking_augmentation")
    x = torch.randn(4, 37, 14)
    x_masked, keep_steps, keep_ratio = temporal_masking_augmentation(x, min_ratio=0.3, max_ratio=0.8)
    print(f"Original shape: {x.shape}")
    print(f"Masked shape: {x_masked.shape}")
    print(f"Keep steps: {keep_steps} ({keep_ratio*100:.1f}%)")
    print(f"Non-zero steps: {(x_masked.abs().sum(dim=(0,2)) > 0).sum().item()}")
    
    # 测试课程学习
    print("\n测试CurriculumScheduler")
    scheduler = CurriculumScheduler(total_epochs=100, warmup_epochs=20)
    
    for epoch in [0, 10, 20, 40, 60, 80, 100]:
        min_r, max_r = scheduler.get_time_range(epoch)
        print(f"Epoch {epoch:3d}: time_range=[{min_r:.2f}, {max_r:.2f}]")

