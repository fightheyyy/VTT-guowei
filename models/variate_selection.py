"""
变量选择模块
使用交叉注意力动态选择对预测最重要的变量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariateSelection(nn.Module):
    """
    变量选择模块：使用交叉注意力增强变量表示
    Query: CLS_text（已对齐的语言特征）
    Key & Value: H（原始时间序列的粗粒度编码）
    """
    
    def __init__(self, d_model=512, n_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 多头交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, cls_text, H):
        """
        前向传播
        输入:
            - cls_text: [Batch, N_Variates, D_Model] 查询
            - H: [Batch, N_Variates, D_Model] 键和值
        输出:
            - v_CLS: [Batch, N_Variates, D_Model] 增强的变量表示
        """
        # 交叉注意力
        # Query: cls_text, Key & Value: H
        attn_output, _ = self.cross_attention(
            query=cls_text,
            key=H,
            value=H
        )  # [Batch, N_Variates, D_Model]
        
        # 残差连接 + LayerNorm
        cls_text = self.norm1(cls_text + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(cls_text)
        
        # 残差连接 + LayerNorm
        v_CLS = self.norm2(cls_text + ffn_output)
        
        return v_CLS


class VariateEncoder(nn.Module):
    """
    变量编码器：将每个变量的完整时间序列编码为单个向量H
    这个编码器会处理原始时间序列数据
    """
    
    def __init__(self, d_model=512, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影层（将时间步映射到d_model维度）
        # 这里我们使用平均池化后的线性投影
        self.input_projection = nn.Linear(1, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出投影（全局池化）
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        前向传播
        输入: x [Batch, Time_Steps, N_Variates]
        输出: H [Batch, N_Variates, D_Model]
        """
        batch_size, time_steps, n_variates = x.shape
        
        # 转置为 [Batch, N_Variates, Time_Steps]
        x = x.transpose(1, 2)
        
        # 对每个变量分别处理
        H_list = []
        for v in range(n_variates):
            # 获取单个变量的时间序列 [Batch, Time_Steps]
            x_v = x[:, v, :].unsqueeze(-1)  # [Batch, Time_Steps, 1]
            
            # 投影到d_model维度
            x_v_emb = self.input_projection(x_v)  # [Batch, Time_Steps, D_Model]
            
            # 通过Transformer编码器
            encoded = self.transformer_encoder(x_v_emb)  # [Batch, Time_Steps, D_Model]
            
            # 全局平均池化得到单个向量
            # 转置后池化: [Batch, D_Model, Time_Steps] -> [Batch, D_Model, 1]
            pooled = self.pooling(encoded.transpose(1, 2)).squeeze(-1)  # [Batch, D_Model]
            
            H_list.append(pooled)
        
        # 堆叠所有变量的表示
        H = torch.stack(H_list, dim=1)  # [Batch, N_Variates, D_Model]
        
        return H

