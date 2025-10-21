"""
多模态对齐模块
通过InfoNCE对比学习损失对齐视觉和语言特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignment(nn.Module):
    """
    对比学习对齐模块：使用InfoNCE损失对齐视觉和语言特征
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, vision_features, language_features):
        """
        计算InfoNCE对比损失
        输入:
            - vision_features: [Batch, N_Variates, D_Model]
            - language_features: [Batch, N_Variates, D_Model]
        输出:
            - loss: 标量张量
        """
        batch_size, n_variates, d_model = vision_features.shape
        
        # 重塑为 [Batch * N_Variates, D_Model]
        vision_flat = vision_features.reshape(batch_size * n_variates, d_model)
        language_flat = language_features.reshape(batch_size * n_variates, d_model)
        
        # L2归一化
        vision_flat = F.normalize(vision_flat, p=2, dim=1)
        language_flat = F.normalize(language_flat, p=2, dim=1)
        
        # 计算相似度矩阵 [Batch * N_Variates, Batch * N_Variates]
        # 同一样本同一变量的特征对是正样本对
        logits_v2l = torch.matmul(vision_flat, language_flat.T) / self.temperature
        logits_l2v = logits_v2l.T
        
        # 创建标签：对角线上是正样本对
        labels = torch.arange(batch_size * n_variates, device=vision_features.device)
        
        # 计算交叉熵损失（双向）
        loss_v2l = F.cross_entropy(logits_v2l, labels)
        loss_l2v = F.cross_entropy(logits_l2v, labels)
        
        # 总对比损失
        contrastive_loss = (loss_v2l + loss_l2v) / 2
        
        return contrastive_loss
    
    def get_similarity(self, vision_features, language_features):
        """
        获取视觉和语言特征之间的相似度矩阵（用于分析）
        输入:
            - vision_features: [Batch, N_Variates, D_Model]
            - language_features: [Batch, N_Variates, D_Model]
        输出:
            - similarity: [Batch * N_Variates, Batch * N_Variates]
        """
        batch_size, n_variates, d_model = vision_features.shape
        
        vision_flat = vision_features.reshape(batch_size * n_variates, d_model)
        language_flat = language_features.reshape(batch_size * n_variates, d_model)
        
        vision_flat = F.normalize(vision_flat, p=2, dim=1)
        language_flat = F.normalize(language_flat, p=2, dim=1)
        
        similarity = torch.matmul(vision_flat, language_flat.T)
        
        return similarity

