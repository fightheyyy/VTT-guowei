"""
语言模块
使用Transformer处理数值补丁序列
"""

import torch
import torch.nn as nn
from transformers import CLIPModel


class LanguageModule(nn.Module):
    """
    语言模块：将数值patch序列转换为语言特征
    输入: patches [Batch, N_Variates, N_Patches, Patch_Length]
    输出: 
        - CLS_text [Batch, N_Variates, D_Model] 用于对齐
        - Feat_text [Batch, N_Variates, N_Patches + 1, D_Model] 用于预测
    """
    
    def __init__(self, patch_length, d_model=512, n_heads=8, n_layers=6, dropout=0.1, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.patch_length = patch_length
        self.d_model = d_model
        
        # Tokenizer: 将数值patch映射到嵌入空间
        self.tokenizer = nn.Linear(patch_length, d_model)
        
        # 使用标准Transformer编码器（可训练）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 可学习的[CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 位置编码（可学习）
        max_seq_len = 1024
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, patches):
        """
        前向传播
        输入: patches [Batch, N_Variates, N_Patches, Patch_Length]
        输出:
            - CLS_text [Batch, N_Variates, D_Model]
            - Feat_text [Batch, N_Variates, N_Patches + 1, D_Model]
        """
        batch_size, n_variates, n_patches, patch_length = patches.shape
        
        # 重塑为 [Batch * N_Variates, N_Patches, Patch_Length]
        patches_flat = patches.reshape(batch_size * n_variates, n_patches, patch_length)
        
        # Tokenize: 将每个patch映射到嵌入空间
        patch_embeddings = self.tokenizer(patches_flat)  # [Batch * N_Variates, N_Patches, D_Model]
        
        # 添加[CLS] token到序列开头
        cls_tokens = self.cls_token.expand(batch_size * n_variates, -1, -1)  # [Batch * N_Variates, 1, D_Model]
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)  # [Batch * N_Variates, N_Patches + 1, D_Model]
        
        # 添加位置编码
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.position_embeddings[:, :seq_len, :]
        
        # LayerNorm
        embeddings = self.layer_norm(embeddings)
        
        # 通过Transformer编码器
        hidden_states = self.transformer_encoder(embeddings)  # [Batch * N_Variates, N_Patches + 1, D_Model]
        
        # 提取[CLS] token（第一个位置）用于对齐
        CLS_text = hidden_states[:, 0, :]  # [Batch * N_Variates, D_Model]
        
        # 重塑回原始shape
        CLS_text = CLS_text.reshape(batch_size, n_variates, self.d_model)
        Feat_text = hidden_states.reshape(batch_size, n_variates, seq_len, self.d_model)
        
        return CLS_text, Feat_text

