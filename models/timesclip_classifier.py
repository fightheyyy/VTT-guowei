"""
TimesCLIP分类器 - 基于TimesCLIP论文的分类模型
用于时间序列分类任务
"""

import torch
import torch.nn as nn
from models.language_module_clip import LanguageModuleCLIP
from models.vision_module import VisionModule
from models.preprocessor import LanguagePreprocessor, VisualPreprocessor
from models.contrastive_loss import InfoNCELoss
from models.variate_selection_timesclip import VariateSelectionModule


class TimesCLIPClassifier(nn.Module):
    """
    TimesCLIP分类模型
    结合视觉和语言模态进行时间序列分类
    """
    
    def __init__(
        self,
        time_steps=37,
        n_variates=14,
        num_classes=4,
        d_model=512,
        patch_length=4,
        stride=4,
        clip_model_name="openai/clip-vit-base-patch16",
        use_variate_selection=True,
        use_contrastive=True,
        dropout=0.1
    ):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_variates = n_variates
        self.num_classes = num_classes
        self.use_variate_selection = use_variate_selection
        self.use_contrastive = use_contrastive
        
        # 预处理器
        self.visual_preprocessor = VisualPreprocessor(
            image_size=224
        )
        
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        
        # 双模态编码器
        # 注意：分类任务中，不聚合变量，而是通过变量选择模块处理
        self.vision_module = VisionModule(
            d_model=d_model,
            clip_model_name=clip_model_name,
            aggregate_variates=False  # 保留变量维度，后续通过变量选择或池化处理
        )
        
        self.language_module = LanguageModuleCLIP(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name,
            freeze_backbone=True
        )
        
        # 获取特征维度
        vision_dim = self.vision_module.d_model  # 512
        language_dim = self.language_module.d_model  # 512
        
        # 变量选择模块（或简单平均池化）
        if use_variate_selection:
            self.variate_selection = VariateSelectionModule(
                d_model=d_model,
                n_heads=8,
                dropout=dropout
            )
            # 不展平，而是平均池化
            selected_dim = d_model
        else:
            self.variate_selection = None
            selected_dim = 0
        
        # 对比学习损失
        if use_contrastive:
            self.contrastive_loss_fn = InfoNCELoss(temperature=0.07)
        else:
            self.contrastive_loss_fn = None
        
        # 特征融合维度
        fusion_dim = vision_dim + language_dim + selected_dim
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        print(f"\n[TimesCLIPClassifier] 初始化完成:")
        print(f"  时间步数: {time_steps}")
        print(f"  变量数: {n_variates}")
        print(f"  类别数: {num_classes}")
        print(f"  视觉特征维度: {vision_dim}")
        print(f"  语言特征维度: {language_dim}")
        print(f"  变量选择: {'是' if use_variate_selection else '否'}")
        print(f"  对比学习: {'是' if use_contrastive else '否'}")
        print(f"  融合维度: {fusion_dim}")
        
    def forward(self, x, return_contrastive_loss=False, cached_images=None):
        """
        前向传播
        
        Args:
            x: (batch_size, time_steps, n_variates)
            return_contrastive_loss: 是否返回对比学习损失
            cached_images: (batch_size, n_variates, 3, 224, 224) - 预缓存的图像（可选）
            
        Returns:
            logits: (batch_size, num_classes)
            contrastive_loss (可选): 对比学习损失
        """
        batch_size = x.size(0)
        
        # 1. 视觉分支
        if cached_images is not None:
            # 使用预缓存的图像，跳过matplotlib绘图
            images = cached_images
        else:
            images = self.visual_preprocessor(x)  # (batch_size, n_variates, 3, 224, 224)
        CLS_img_variates = self.vision_module(images)  # (batch_size, n_variates, d_model)
        
        # 2. 语言分支
        patches = self.language_preprocessor(x)  # (batch_size, n_variates, n_patches, patch_length)
        CLS_text_variates, _ = self.language_module(patches)  # (batch_size, n_variates, d_model)
        
        # 3. 特征聚合（平均池化）
        CLS_img = CLS_img_variates.mean(dim=1)  # (batch_size, d_model)
        CLS_text = CLS_text_variates.mean(dim=1)  # (batch_size, d_model)
        
        # 4. 对比学习损失（如果需要）
        contrastive_loss = None
        if return_contrastive_loss and self.use_contrastive:
            contrastive_loss = self.contrastive_loss_fn(CLS_img, CLS_text)
        
        # 5. 变量选择（可选）
        features_list = [CLS_img, CLS_text]
        if self.use_variate_selection:
            # 使用未聚合的变量级特征进行选择
            selected_features = self.variate_selection(CLS_img_variates, CLS_text_variates)
            # 平均池化选中的特征
            selected_features = selected_features.mean(dim=1)  # (batch_size, d_model)
            features_list.append(selected_features)
        
        # 6. 特征融合
        fused_features = torch.cat(features_list, dim=1)  # (batch_size, fusion_dim)
        
        # 7. 分类
        logits = self.classifier(fused_features)  # (batch_size, num_classes)
        
        if return_contrastive_loss and contrastive_loss is not None:
            return logits, contrastive_loss
        else:
            return logits
    
    def compute_loss(self, x, y, contrastive_weight=0.1, cached_images=None):
        """
        计算联合损失（交叉熵 + 对比学习）
        
        Args:
            x: (batch_size, time_steps, n_variates)
            y: (batch_size,) - 类别标签
            contrastive_weight: 对比学习损失权重
            cached_images: (batch_size, n_variates, 3, 224, 224) - 预缓存的图像（可选）
            
        Returns:
            total_loss: 总损失
            loss_dict: 包含各项损失的字典
        """
        # 前向传播
        if self.use_contrastive:
            logits, contrastive_loss = self.forward(x, return_contrastive_loss=True, cached_images=cached_images)
        else:
            logits = self.forward(x, return_contrastive_loss=False, cached_images=cached_images)
            contrastive_loss = torch.tensor(0.0, device=x.device)
        
        # 分类损失（交叉熵）
        ce_loss = nn.functional.cross_entropy(logits, y)
        
        # 总损失
        total_loss = ce_loss + contrastive_weight * contrastive_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'contrastive': contrastive_loss.item(),
            'logits': logits.detach()  # 返回logits，避免重复forward
        }
        
        return total_loss, loss_dict


class LanguageOnlyTimesCLIPClassifier(nn.Module):
    """
    纯语言模态的TimesCLIP分类器（用于对比实验）
    """
    
    def __init__(
        self,
        time_steps=37,
        n_variates=14,
        num_classes=4,
        d_model=512,
        patch_length=4,
        stride=4,
        clip_model_name="openai/clip-vit-base-patch16",
        dropout=0.1
    ):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_variates = n_variates
        self.num_classes = num_classes
        
        # 预处理器
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        
        # 语言模块
        self.language_module = LanguageModuleCLIP(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name,
            freeze_backbone=True
        )
        
        language_dim = self.language_module.d_model  # 512
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(language_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        print(f"\n[LanguageOnlyTimesCLIPClassifier] 初始化完成:")
        print(f"  时间步数: {time_steps}")
        print(f"  变量数: {n_variates}")
        print(f"  类别数: {num_classes}")
        print(f"  特征维度: {language_dim}")
        
    def forward(self, x, cached_images=None):
        """
        前向传播
        
        Args:
            x: (batch_size, time_steps, n_variates)
            cached_images: 兼容参数（纯语言模型不使用）
            
        Returns:
            logits: (batch_size, num_classes)
        """
        # 语言分支
        patches = self.language_preprocessor(x)
        CLS_text, _ = self.language_module(patches)
        
        # 分类
        logits = self.classifier(CLS_text)
        
        return logits
    
    def compute_loss(self, x, y, cached_images=None):
        """
        计算交叉熵损失
        
        Args:
            cached_images: 兼容参数（纯语言模型不使用）
        """
        logits = self.forward(x, cached_images=cached_images)
        ce_loss = nn.functional.cross_entropy(logits, y)
        
        loss_dict = {
            'total': ce_loss.item(),
            'ce': ce_loss.item(),
            'logits': logits.detach()  # 返回logits，避免重复forward
        }
        
        return ce_loss, loss_dict


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 8
    time_steps = 37
    n_variates = 14
    num_classes = 4
    
    x = torch.randn(batch_size, time_steps, n_variates).to(device)
    y = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 测试双模态模型
    print("测试双模态模型...")
    model = TimesCLIPClassifier(
        time_steps=time_steps,
        n_variates=n_variates,
        num_classes=num_classes
    ).to(device)
    
    logits, contrastive_loss = model(x, return_contrastive_loss=True)
    print(f"logits.shape: {logits.shape}")
    print(f"contrastive_loss: {contrastive_loss.item():.4f}")
    
    loss, loss_dict = model.compute_loss(x, y)
    print(f"loss_dict: {loss_dict}")
    
    # 测试纯语言模型
    print("\n测试纯语言模型...")
    model_lang = LanguageOnlyTimesCLIPClassifier(
        time_steps=time_steps,
        n_variates=n_variates,
        num_classes=num_classes
    ).to(device)
    
    logits = model_lang(x)
    print(f"logits.shape: {logits.shape}")
    
    loss, loss_dict = model_lang.compute_loss(x, y)
    print(f"loss_dict: {loss_dict}")

