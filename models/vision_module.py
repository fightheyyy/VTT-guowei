"""
视觉模块
使用预训练CLIP ViT提取视觉特征
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class VisionModule(nn.Module):
    """
    视觉模块：使用预训练的CLIP ViT-B/16提取图像特征
    输入: images [Batch * N_Variates, 3, 224, 224]
    输出: CLS_img [Batch, N_Variates, D_Model]
    """
    
    def __init__(self, d_model=512, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.d_model = d_model
        
        # 加载预训练的CLIP Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        
        # 冻结预训练参数
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # CLIP ViT-B/16的输出维度是768
        clip_output_dim = self.vision_encoder.config.hidden_size
        
        # 可训练的多模态投影层
        self.projection = nn.Sequential(
            nn.Linear(clip_output_dim, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, images):
        """
        前向传播
        输入: images [Batch, N_Variates, 3, H, W]
        输出: CLS_img [Batch, N_Variates, D_Model]
        """
        batch_size, n_variates, c, h, w = images.shape
        
        # 重塑为 [Batch * N_Variates, 3, H, W]
        images_flat = images.reshape(batch_size * n_variates, c, h, w)
        
        # 通过CLIP Vision Encoder提取特征
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=images_flat)
            # 使用pooler_output（相当于[CLS] token的输出）
            vision_features = outputs.pooler_output  # [Batch * N_Variates, 768]
        
        # 通过可训练的投影层
        vision_features = self.projection(vision_features)  # [Batch * N_Variates, D_Model]
        
        # 重塑回 [Batch, N_Variates, D_Model]
        CLS_img = vision_features.reshape(batch_size, n_variates, self.d_model)
        
        return CLS_img

