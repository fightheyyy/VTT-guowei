"""
语言模块 - CLIP-Text版本
使用预训练的CLIP Text Encoder作为backbone
完全对齐TimesCLIP论文
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class LanguageModuleCLIP(nn.Module):
    """
    使用CLIP-Text作为backbone的语言模块
    
    论文核心观点：
    "CLIP-Text as backbone 真的很有用，因为其 feature space 是多模态空间的，
     同时具有了 language 的特性和 vision 的特性"
    
    输入: patches [Batch, N_Variates, N_Patches, Patch_Length]
    输出:
        - CLS_text [Batch, N_Variates, D_Model] 用于对齐
        - Feat_text [Batch, N_Variates, N_Patches + 1, D_Model] 用于预测
    """
    
    def __init__(
        self, 
        patch_length, 
        d_model=512, 
        clip_model_name="openai/clip-vit-base-patch16",
        freeze_backbone=True,
        freeze_layers=None
    ):
        super().__init__()
        self.patch_length = patch_length
        self.d_model = d_model
        
        # 加载预训练的CLIP Text Encoder
        clip_model = CLIPTextModel.from_pretrained(
            clip_model_name,
            local_files_only=False  # 允许在线下载
        )
        
        # 直接使用encoder部分（transformer层）
        # CLIPTextTransformer.encoder 是真正的transformer编码器
        self.text_encoder = clip_model.text_model.encoder
        self.text_final_layer_norm = clip_model.text_model.final_layer_norm
        
        # CLIP Text的隐藏维度（ViT-B/16: 512）
        clip_hidden_size = clip_model.config.hidden_size
        self.clip_hidden_size = clip_hidden_size
        
        # 冻结策略
        if freeze_backbone:
            if freeze_layers is None:
                # 完全冻结encoder和final_layer_norm
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                for param in self.text_final_layer_norm.parameters():
                    param.requires_grad = False
                print(f"[LanguageModuleCLIP] CLIP-Text Encoder完全冻结")
            else:
                # 只冻结前N层
                for i, layer in enumerate(self.text_encoder.layers):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"[LanguageModuleCLIP] 冻结前{freeze_layers}层，微调后{len(self.text_encoder.layers) - freeze_layers}层")
        else:
            print(f"[LanguageModuleCLIP] CLIP-Text Encoder完全可训练")
        
        # Patch tokenizer: 将数值patch映射到CLIP的输入空间
        self.patch_tokenizer = nn.Sequential(
            nn.Linear(patch_length, clip_hidden_size),
            nn.LayerNorm(clip_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # [CLS] token（可学习）
        self.cls_token = nn.Parameter(torch.randn(1, 1, clip_hidden_size))
        
        # 位置编码（可学习）
        max_seq_len = 1024
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, clip_hidden_size))
        
        # 投影层：将CLIP输出映射到目标维度
        if clip_hidden_size != d_model:
            self.projection = nn.Sequential(
                nn.Linear(clip_hidden_size, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            self.projection = nn.Identity()
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(clip_hidden_size)
    
    def forward(self, patches):
        """
        前向传播
        
        输入: patches [Batch, N_Variates, N_Patches, Patch_Length]
        输出:
            - CLS_text [Batch, N_Variates, D_Model] - 用于对齐和分类
            - Feat_text [Batch, N_Variates, N_Patches + 1, D_Model] - 用于生成
        """
        batch_size, n_variates, n_patches, patch_length = patches.shape
        
        # 重塑为 [Batch * N_Variates, N_Patches, Patch_Length]
        patches_flat = patches.reshape(batch_size * n_variates, n_patches, patch_length)
        
        # Tokenize: 将每个patch映射到CLIP的嵌入空间
        patch_embeddings = self.patch_tokenizer(patches_flat)  # [B*V, N_Patches, clip_hidden]
        
        # 添加[CLS] token到序列开头
        cls_tokens = self.cls_token.expand(batch_size * n_variates, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)  # [B*V, N_Patches+1, clip_hidden]
        
        # 添加位置编码
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.position_embeddings[:, :seq_len, :]
        
        # LayerNorm
        embeddings = self.layer_norm(embeddings)
        
        # 通过CLIP Text Encoder
        # 直接调用encoder（transformer层），传入embeddings
        # 创建causal attention mask（CLIP文本使用causal attention）
        # 但对于我们的时序任务，使用双向注意力（全1 mask）
        attention_mask = torch.ones(
            embeddings.shape[0], embeddings.shape[1], 
            dtype=torch.long, device=embeddings.device
        )
        
        # 扩展attention_mask维度以匹配multi-head attention
        # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embeddings.dtype).min
        
        # 通过CLIP的transformer encoder
        hidden_states = self.text_encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            causal_attention_mask=None,  # 不使用causal mask
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )[0]  # 返回的是tuple，取第一个元素
        
        # 通过final layer norm
        hidden_states = self.text_final_layer_norm(hidden_states)  # [B*V, N_Patches+1, clip_hidden]
        
        # 提取[CLS] token
        cls_features = hidden_states[:, 0, :]  # [B*V, clip_hidden]
        
        # 投影到目标维度
        cls_features = self.projection(cls_features)  # [B*V, d_model]
        hidden_states_proj = self.projection(hidden_states)  # [B*V, N_Patches+1, d_model]
        
        # 重塑回原始shape
        CLS_text = cls_features.reshape(batch_size, n_variates, self.d_model)
        Feat_text = hidden_states_proj.reshape(batch_size, n_variates, seq_len, self.d_model)
        
        return CLS_text, Feat_text


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("测试 CLIP-Text 语言模块")
    print("="*60)
    
    batch_size = 2
    n_variates = 7
    n_patches = 3
    patch_length = 6
    
    # 创建模型
    model = LanguageModuleCLIP(
        patch_length=patch_length,
        d_model=256,
        freeze_backbone=True
    )
    
    # 测试输入
    patches = torch.randn(batch_size, n_variates, n_patches, patch_length)
    
    # 前向传播
    CLS_text, Feat_text = model(patches)
    
    print(f"\n输入: {patches.shape}")
    print(f"CLS_text: {CLS_text.shape}")
    print(f"Feat_text: {Feat_text.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"冻结参数: {total_params - trainable_params:,}")

