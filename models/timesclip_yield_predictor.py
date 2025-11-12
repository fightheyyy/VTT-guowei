"""
TimesCLIP产量预测模型
完全对齐论文方法：
1. 使用CLIP-Text作为语言backbone
2. 使用CLIP-Vision作为视觉backbone
3. 多模态对比学习
4. 变量选择模块
"""

import torch
import torch.nn as nn
from .vision_module import VisionModule
from .language_module_clip import LanguageModuleCLIP
from .preprocessor import VisualPreprocessor, LanguagePreprocessor
from .variate_selection_timesclip import VariateSelectionModule
from .contrastive_loss import HybridContrastiveLoss


class TimesCLIPYieldPredictor(nn.Module):
    """
    TimesCLIP产量预测器
    
    论文核心方法：
    1. CLIP-Text + CLIP-Vision双模态
    2. 多模态对比学习（InfoNCE）
    3. 变量选择模块
    4. 端到端训练
    
    输入: [Batch, Time_Steps, N_Variates]
    输出: [Batch, 1] 产量预测
    """
    
    def __init__(
        self,
        time_steps=18,
        n_variates=7,
        d_model=256,
        patch_length=6,
        stride=3,
        clip_model_name="openai/clip-vit-base-patch16",
        use_variate_selection=True,
        freeze_clip_text=True,
        freeze_clip_vision=True,
        contrastive_weight=0.1
    ):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_variates = n_variates
        self.d_model = d_model
        self.use_variate_selection = use_variate_selection
        self.contrastive_weight = contrastive_weight
        
        # ===== 视觉分支 =====
        self.visual_preprocessor = VisualPreprocessor(image_size=224)
        self.vision_module = VisionModule(
            d_model=d_model,
            clip_model_name=clip_model_name
        )
        # CLIP-Vision默认冻结
        
        # ===== 语言分支（CLIP-Text）=====
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        self.language_module = LanguageModuleCLIP(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name,
            freeze_backbone=freeze_clip_text
        )
        
        # ===== 变量选择模块 =====
        if use_variate_selection:
            self.variate_selection = VariateSelectionModule(
                d_model=d_model,
                n_heads=8,
                dropout=0.1
            )
        
        # ===== 对比学习损失 =====
        self.contrastive_loss_fn = HybridContrastiveLoss(
            temperature=0.07,
            alpha=0.5
        )
        
        # ===== 回归头 =====
        # 融合维度：视觉 + 语言 (+ 变量选择)
        if use_variate_selection:
            fusion_dim = d_model * n_variates * 3  # vision + language + selected
        else:
            fusion_dim = d_model * n_variates * 2  # vision + language
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x, return_contrastive_loss=False):
        """
        前向传播
        
        输入:
            x: [Batch, Time_Steps, N_Variates]
            return_contrastive_loss: bool - 是否返回对比损失
        
        输出:
            yield_pred: [Batch, 1]
            contrastive_loss: 标量 (可选)
        """
        batch_size = x.shape[0]
        
        # ===== 视觉分支 =====
        images = self.visual_preprocessor(x)  # [B, V, 3, 224, 224]
        CLS_img = self.vision_module(images)  # [B, V, D]
        
        # ===== 语言分支 =====
        patches = self.language_preprocessor(x)  # [B, V, N_Patches, Patch_Len]
        CLS_text, Feat_text = self.language_module(patches)  # [B, V, D], [B, V, N+1, D]
        
        # ===== 对比学习损失 =====
        contrastive_loss = None
        if return_contrastive_loss:
            contrastive_loss, _ = self.contrastive_loss_fn(CLS_img, CLS_text)
        
        # ===== 变量选择 =====
        features_to_fuse = [
            CLS_img.reshape(batch_size, -1),
            CLS_text.reshape(batch_size, -1)
        ]
        
        if self.use_variate_selection:
            selected_features = self.variate_selection(CLS_img, CLS_text)
            features_to_fuse.append(selected_features.reshape(batch_size, -1))
        
        # ===== 特征融合 =====
        fused = torch.cat(features_to_fuse, dim=1)  # [B, Fusion_Dim]
        
        # ===== 回归预测 =====
        yield_pred = self.regressor(fused)  # [B, 1]
        
        if return_contrastive_loss:
            return yield_pred, contrastive_loss
        else:
            return yield_pred
    
    def compute_loss(self, x, y):
        """
        计算总损失：回归损失 + 对比学习损失
        
        输入:
            x: [Batch, Time_Steps, N_Variates]
            y: [Batch, 1]
        
        输出:
            total_loss: 标量
            loss_dict: 损失详情字典
        """
        # 前向传播
        yield_pred, contrastive_loss = self.forward(x, return_contrastive_loss=True)
        
        # 回归损失
        regression_loss = nn.functional.mse_loss(yield_pred, y)
        
        # 总损失
        total_loss = regression_loss + self.contrastive_weight * contrastive_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'regression_loss': regression_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'contrastive_weight': self.contrastive_weight
        }
        
        return total_loss, loss_dict


class LanguageOnlyTimesCLIPPredictor(nn.Module):
    """
    纯语言模态版本（使用CLIP-Text）
    用于消融实验
    """
    
    def __init__(
        self,
        time_steps=18,
        n_variates=7,
        d_model=256,
        patch_length=6,
        stride=3,
        clip_model_name="openai/clip-vit-base-patch16",
        freeze_clip_text=True
    ):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_variates = n_variates
        self.d_model = d_model
        
        # 语言分支（CLIP-Text）
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        self.language_module = LanguageModuleCLIP(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name,
            freeze_backbone=freeze_clip_text
        )
        
        # 回归头
        fusion_dim = d_model * n_variates
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 语言分支
        patches = self.language_preprocessor(x)
        CLS_text, _ = self.language_module(patches)
        
        # 展平
        fused = CLS_text.reshape(batch_size, -1)
        
        # 回归
        yield_pred = self.regressor(fused)
        
        return yield_pred


if __name__ == "__main__":
    print("="*70)
    print("测试 TimesCLIP 产量预测模型")
    print("="*70)
    
    batch_size = 4
    time_steps = 12
    n_variates = 7
    
    x = torch.randn(batch_size, time_steps, n_variates)
    y = torch.randn(batch_size, 1)
    
    # 完整模型
    print("\n[1] TimesCLIP完整模型（双模态 + 对比学习 + 变量选择）")
    model = TimesCLIPYieldPredictor(
        time_steps=time_steps,
        n_variates=n_variates,
        d_model=256,
        use_variate_selection=True,
        contrastive_weight=0.1
    )
    
    # 前向传播
    yield_pred = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {yield_pred.shape}")
    
    # 计算损失
    total_loss, loss_dict = model.compute_loss(x, y)
    print(f"\n损失详情:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  冻结: {total_params - trainable_params:,}")
    
    # 纯语言模型
    print("\n" + "="*70)
    print("[2] 纯语言模态（CLIP-Text only）")
    model_lang = LanguageOnlyTimesCLIPPredictor(
        time_steps=time_steps,
        n_variates=n_variates,
        d_model=256
    )
    
    yield_pred = model_lang(x)
    print(f"输入: {x.shape}")
    print(f"输出: {yield_pred.shape}")
    
    total_params = sum(p.numel() for p in model_lang.parameters())
    trainable_params = sum(p.numel() for p in model_lang.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

