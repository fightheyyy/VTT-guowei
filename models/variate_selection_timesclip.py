"""
变量选择模块 - TimesCLIP版本
通过对比学习学习变量间关系，选择重要变量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariateSelectionModule(nn.Module):
    """
    变量选择模块
    
    论文核心思想：
    - 引入可学习的[CLS]token来表示每个变量
    - 通过视觉和语言的对比学习约束这些CLS token
    - 计算变量间相似度，选择重要变量
    - 用选中的变量指导最终生成
    
    输入:
        CLS_img: [Batch, N_Variates, D] - 视觉特征
        CLS_text: [Batch, N_Variates, D] - 语言特征
    
    输出:
        selection_weights: [Batch, N_Variates, N_Variates] - 变量选择权重
        selected_features: [Batch, N_Variates, D] - 选择后的特征
    """
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 多头注意力：计算变量间关系
        self.cross_variate_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # 门控机制：决定是否使用跨变量信息
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, CLS_img, CLS_text, return_weights=False):
        """
        前向传播
        
        输入:
            CLS_img: [Batch, N_Variates, D]
            CLS_text: [Batch, N_Variates, D]
            return_weights: bool - 是否返回注意力权重
        
        输出:
            selected_features: [Batch, N_Variates, D]
            attention_weights: [Batch, N_Variates, N_Variates] (可选)
        """
        batch_size, n_variates, d_model = CLS_img.shape
        
        # 融合视觉和语言特征（简单相加）
        fused_features = CLS_img + CLS_text  # [B, V, D]
        
        # 跨变量注意力：每个变量关注其他变量
        attn_output, attn_weights = self.cross_variate_attention(
            query=fused_features,
            key=fused_features,
            value=fused_features,
            need_weights=True,
            average_attn_weights=True
        )
        
        # 残差连接 + LayerNorm
        fused_features = self.norm1(fused_features + attn_output)
        
        # FFN + 残差
        ffn_output = self.ffn(fused_features)
        fused_features = self.norm2(fused_features + ffn_output)
        
        # 门控融合：原始特征 vs 跨变量特征
        # 拼接原始和增强后的特征
        concat_features = torch.cat([CLS_img + CLS_text, fused_features], dim=-1)
        gate_values = self.gate(concat_features)  # [B, V, D]
        
        # 门控组合
        selected_features = gate_values * fused_features + (1 - gate_values) * (CLS_img + CLS_text)
        
        if return_weights:
            return selected_features, attn_weights
        else:
            return selected_features


class TopKVariateSelection(nn.Module):
    """
    Top-K变量选择
    
    基于相似度选择最重要的K个变量
    适用于高维多变量时间序列
    """
    
    def __init__(self, d_model, top_k=None, temperature=0.1):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.temperature = temperature
        
        # 重要性评分网络
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, CLS_img, CLS_text, k=None):
        """
        选择Top-K重要变量
        
        输入:
            CLS_img: [Batch, N_Variates, D]
            CLS_text: [Batch, N_Variates, D]
            k: int - 选择的变量数量（None则使用self.top_k）
        
        输出:
            selected_features: [Batch, K, D] - 选中的特征
            selected_indices: [Batch, K] - 选中的索引
            importance_scores: [Batch, N_Variates] - 重要性分数
        """
        batch_size, n_variates, d_model = CLS_img.shape
        k = k or self.top_k or n_variates
        
        # 融合视觉和语言特征
        fused_features = (CLS_img + CLS_text) / 2.0
        
        # 计算重要性分数
        importance_scores = self.importance_scorer(fused_features).squeeze(-1)  # [B, V]
        
        # Softmax归一化（使用temperature控制平滑度）
        importance_probs = F.softmax(importance_scores / self.temperature, dim=-1)
        
        # 选择Top-K
        topk_probs, topk_indices = torch.topk(importance_probs, k=min(k, n_variates), dim=-1)
        
        # 根据索引选择特征
        batch_indices = torch.arange(batch_size, device=CLS_img.device).unsqueeze(1).expand(-1, k)
        selected_features = fused_features[batch_indices, topk_indices]  # [B, K, D]
        
        return selected_features, topk_indices, importance_probs


if __name__ == "__main__":
    print("="*60)
    print("测试变量选择模块")
    print("="*60)
    
    batch_size = 4
    n_variates = 7
    d_model = 256
    
    CLS_img = torch.randn(batch_size, n_variates, d_model)
    CLS_text = torch.randn(batch_size, n_variates, d_model)
    
    # 测试基础变量选择
    print("\n[1] 基础变量选择模块")
    module = VariateSelectionModule(d_model=d_model, n_heads=8)
    selected_features, attn_weights = module(CLS_img, CLS_text, return_weights=True)
    
    print(f"输入: {CLS_img.shape}")
    print(f"输出: {selected_features.shape}")
    print(f"注意力权重: {attn_weights.shape}")
    print(f"注意力权重示例（第1个样本）:\n{attn_weights[0].detach().numpy()}")
    
    # 测试Top-K选择
    print("\n[2] Top-K变量选择")
    module = TopKVariateSelection(d_model=d_model, top_k=3)
    selected_features, selected_indices, importance_scores = module(CLS_img, CLS_text, k=3)
    
    print(f"输入: {CLS_img.shape}")
    print(f"选中特征: {selected_features.shape}")
    print(f"选中索引: {selected_indices.shape}")
    print(f"重要性分数: {importance_scores.shape}")
    print(f"\n第1个样本的重要性分数: {importance_scores[0].detach().numpy()}")
    print(f"第1个样本选中的变量索引: {selected_indices[0].detach().numpy()}")
    
    # 参数统计
    total_params = sum(p.numel() for p in module.parameters())
    print(f"\n模型参数量: {total_params:,}")

