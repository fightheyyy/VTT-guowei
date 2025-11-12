"""
对比学习损失模块
实现InfoNCE损失（CLIP风格）
对齐TimesCLIP论文的多模态对比学习策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE损失（CLIP风格的对比学习）
    
    论文核心：
    "多模态对比学习框架 (Multimodal Contrastive Learning) is ALL you NEED"
    
    通过对比学习让视觉特征和语言特征在同一空间对齐
    """
    
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features_a, features_b):
        """
        计算两组特征之间的InfoNCE损失
        
        输入:
            features_a: [Batch, D] - 第一组特征（如视觉）
            features_b: [Batch, D] - 第二组特征（如语言）
        
        输出:
            loss: 标量 - 对比损失
        """
        batch_size = features_a.size(0)
        
        # L2归一化
        features_a = F.normalize(features_a, p=2, dim=-1)
        features_b = F.normalize(features_b, p=2, dim=-1)
        
        # 计算相似度矩阵
        logits_a_to_b = torch.matmul(features_a, features_b.mT) / self.temperature  # [B, B]
        logits_b_to_a = logits_a_to_b.mT  # [B, B]
        
        # 标签：对角线为正样本
        labels = torch.arange(batch_size, device=features_a.device)
        
        # 双向对比损失
        loss_a_to_b = F.cross_entropy(logits_a_to_b, labels, reduction=self.reduction)
        loss_b_to_a = F.cross_entropy(logits_b_to_a, labels, reduction=self.reduction)
        
        # 平均
        loss = (loss_a_to_b + loss_b_to_a) / 2.0
        
        return loss


class MultiVariateContrastiveLoss(nn.Module):
    """
    多变量对比学习损失
    
    对每个变量的视觉特征和语言特征进行对齐
    适用于时间序列的多变量场景
    """
    
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.info_nce = InfoNCELoss(temperature, reduction='none')
    
    def forward(self, CLS_img, CLS_text):
        """
        计算多变量的对比损失
        
        输入:
            CLS_img: [Batch, N_Variates, D] - 视觉CLS特征
            CLS_text: [Batch, N_Variates, D] - 语言CLS特征
        
        输出:
            loss: 标量 - 平均对比损失
        """
        batch_size, n_variates, d_model = CLS_img.shape
        
        # 方法1：每个变量独立计算（变量内对齐）
        total_loss = 0.0
        for v in range(n_variates):
            loss_v = self.info_nce(CLS_img[:, v, :], CLS_text[:, v, :])
            total_loss += loss_v.mean()
        
        return total_loss / n_variates


class GlobalContrastiveLoss(nn.Module):
    """
    全局对比学习损失
    
    将所有变量的特征拼接后进行对比学习
    捕捉全局的多变量模式
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.info_nce = InfoNCELoss(temperature)
    
    def forward(self, CLS_img, CLS_text):
        """
        计算全局对比损失
        
        输入:
            CLS_img: [Batch, N_Variates, D] - 视觉CLS特征
            CLS_text: [Batch, N_Variates, D] - 语言CLS特征
        
        输出:
            loss: 标量 - 全局对比损失
        """
        batch_size, n_variates, d_model = CLS_img.shape
        
        # 展平所有变量
        img_global = CLS_img.reshape(batch_size, -1)  # [B, V*D]
        text_global = CLS_text.reshape(batch_size, -1)  # [B, V*D]
        
        # 计算全局对比损失
        loss = self.info_nce(img_global, text_global)
        
        return loss


class HybridContrastiveLoss(nn.Module):
    """
    混合对比学习损失
    
    结合变量级和全局级对比学习
    """
    
    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.variate_loss = MultiVariateContrastiveLoss(temperature)
        self.global_loss = GlobalContrastiveLoss(temperature)
    
    def forward(self, CLS_img, CLS_text):
        """
        计算混合对比损失
        
        输入:
            CLS_img: [Batch, N_Variates, D]
            CLS_text: [Batch, N_Variates, D]
        
        输出:
            loss: 标量
            loss_dict: 字典 - 各项损失详情
        """
        # 变量级损失
        loss_variate = self.variate_loss(CLS_img, CLS_text)
        
        # 全局级损失
        loss_global = self.global_loss(CLS_img, CLS_text)
        
        # 加权组合
        loss = self.alpha * loss_variate + (1 - self.alpha) * loss_global
        
        loss_dict = {
            'loss_variate': loss_variate.item(),
            'loss_global': loss_global.item(),
            'loss_total': loss.item()
        }
        
        return loss, loss_dict


if __name__ == "__main__":
    print("="*60)
    print("测试对比学习损失")
    print("="*60)
    
    batch_size = 8
    n_variates = 7
    d_model = 256
    
    # 模拟特征
    CLS_img = torch.randn(batch_size, n_variates, d_model)
    CLS_text = torch.randn(batch_size, n_variates, d_model)
    
    # 测试各种损失
    print("\n[1] InfoNCE损失（单特征）")
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(CLS_img[:, 0, :], CLS_text[:, 0, :])
    print(f"Loss: {loss.item():.4f}")
    
    print("\n[2] 多变量对比损失")
    loss_fn = MultiVariateContrastiveLoss(temperature=0.07)
    loss = loss_fn(CLS_img, CLS_text)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n[3] 全局对比损失")
    loss_fn = GlobalContrastiveLoss(temperature=0.07)
    loss = loss_fn(CLS_img, CLS_text)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n[4] 混合对比损失")
    loss_fn = HybridContrastiveLoss(temperature=0.07, alpha=0.5)
    loss, loss_dict = loss_fn(CLS_img, CLS_text)
    print(f"Variate Loss: {loss_dict['loss_variate']:.4f}")
    print(f"Global Loss: {loss_dict['loss_global']:.4f}")
    print(f"Total Loss: {loss_dict['loss_total']:.4f}")

