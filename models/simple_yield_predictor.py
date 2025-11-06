"""
简单的双模态产量回归模型
输入：前N个时间步（每步10天）的多变量时间序列
输出：产量预测
"""

import torch
import torch.nn as nn
from .vision_module import VisionModule
from .language_module import LanguageModule
from .preprocessor import VisualPreprocessor, LanguagePreprocessor


class SimpleYieldPredictor(nn.Module):
    """
    简单双模态产量预测器
    直接从时间序列预测产量，不做序列补全
    
    参数:
        - time_steps: 输入时间步数（如6步=60天，12步=120天）
        - n_variates: 变量数量（波段数）
        - d_model: 模型隐藏维度
        - use_vision: 是否使用视觉模态
        - use_language: 是否使用语言模态
    """
    
    def __init__(
        self,
        time_steps=18,
        n_variates=7,
        d_model=256,
        use_vision=True,
        use_language=True,
        patch_length=6,
        stride=3,
        clip_model_name="openai/clip-vit-base-patch16"
    ):
        super().__init__()
        
        assert use_vision or use_language, "至少启用一种模态"
        
        self.time_steps = time_steps
        self.n_variates = n_variates
        self.d_model = d_model
        self.use_vision = use_vision
        self.use_language = use_language
        
        # 视觉模态
        if use_vision:
            self.visual_preprocessor = VisualPreprocessor(image_size=224)
            self.vision_module = VisionModule(
                d_model=d_model,
                clip_model_name=clip_model_name
            )
        
        # 语言模态
        if use_language:
            self.language_preprocessor = LanguagePreprocessor(
                patch_length=patch_length,
                stride=stride
            )
            self.language_module = LanguageModule(
                patch_length=patch_length,
                d_model=d_model,
                clip_model_name=clip_model_name
            )
        
        # 融合和回归头
        n_modalities = int(use_vision) + int(use_language)
        fusion_dim = d_model * n_modalities * n_variates
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        输入:
            x: [Batch, Time_Steps, N_Variates] 时间序列
        
        输出:
            yield_pred: [Batch, 1] 产量预测
        """
        batch_size = x.shape[0]
        features = []
        
        # 视觉特征
        if self.use_vision:
            images = self.visual_preprocessor(x)  # [B, N_Variates, 3, 224, 224]
            vision_features = self.vision_module(images)  # [B, N_Variates, D_Model]
            features.append(vision_features.reshape(batch_size, -1))
        
        # 语言特征
        if self.use_language:
            patches = self.language_preprocessor(x)  # [B, N_Variates, N_Patches, Patch_Len]
            language_features, _ = self.language_module(patches)  # [B, N_Variates, D_Model]
            features.append(language_features.reshape(batch_size, -1))
        
        # 融合特征
        fused = torch.cat(features, dim=1)  # [B, Fusion_Dim]
        
        # 回归预测
        yield_pred = self.regressor(fused)  # [B, 1]
        
        return yield_pred


class LanguageOnlyYieldPredictor(nn.Module):
    """
    仅语言模态的产量预测器（更快，参数更少）
    """
    
    def __init__(
        self,
        time_steps=18,
        n_variates=7,
        d_model=256,
        patch_length=6,
        stride=3,
        clip_model_name="openai/clip-vit-base-patch16"
    ):
        super().__init__()
        
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        self.language_module = LanguageModule(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name
        )
        
        fusion_dim = d_model * n_variates
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        patches = self.language_preprocessor(x)
        language_features, _ = self.language_module(patches)
        
        fused = language_features.reshape(batch_size, -1)
        yield_pred = self.regressor(fused)
        
        return yield_pred


if __name__ == "__main__":
    # 测试
    batch_size = 4
    time_steps = 12  # 12步 = 120天
    n_variates = 7
    
    print("="*60)
    print("测试简单产量预测模型")
    print("="*60)
    
    # 双模态
    print("\n[1] 双模态模型")
    model_both = SimpleYieldPredictor(
        time_steps=time_steps,
        n_variates=n_variates,
        d_model=256,
        use_vision=True,
        use_language=True
    )
    
    x = torch.randn(batch_size, time_steps, n_variates)
    y_pred = model_both(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {y_pred.shape}")
    
    total_params = sum(p.numel() for p in model_both.parameters())
    print(f"总参数: {total_params:,}")
    
    # 仅语言模态
    print("\n[2] 仅语言模态")
    model_lang = LanguageOnlyYieldPredictor(
        time_steps=time_steps,
        n_variates=n_variates,
        d_model=256
    )
    
    y_pred = model_lang(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y_pred.shape}")
    
    total_params = sum(p.numel() for p in model_lang.parameters())
    print(f"总参数: {total_params:,}")

