"""
TimesCLIP完整模型
整合所有子模块的主模型类
"""

import torch
import torch.nn as nn
from .preprocessor import VisualPreprocessor, LanguagePreprocessor
from .vision_module import VisionModule
from .language_module import LanguageModule
from .alignment import ContrastiveAlignment
from .variate_selection import VariateSelection, VariateEncoder
from .generator import Generator


class TimesCLIP(nn.Module):
    """
    TimesCLIP: 双模态时间序列预测模型
    
    参数:
        - time_steps: 输入时间序列长度
        - n_variates: 变量数量
        - prediction_steps: 预测步长
        - patch_length: patch长度
        - stride: patch步长
        - d_model: 模型隐藏维度
        - n_heads: 注意力头数
        - temperature: 对比学习温度参数
        - image_size: 图像尺寸
        - clip_model_name: CLIP模型名称
    """
    
    def __init__(
        self,
        time_steps=96,
        n_variates=7,
        prediction_steps=96,
        patch_length=16,
        stride=8,
        d_model=512,
        n_heads=8,
        temperature=0.07,
        image_size=224,
        clip_model_name="openai/clip-vit-base-patch16"
    ):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_variates = n_variates
        self.prediction_steps = prediction_steps
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        
        # 计算patch数量
        self.n_patches = (time_steps - patch_length) // stride + 1
        
        # 1. 数据预处理器
        self.visual_preprocessor = VisualPreprocessor(image_size=image_size)
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        
        # 2. 视觉模块
        self.vision_module = VisionModule(
            d_model=d_model,
            clip_model_name=clip_model_name
        )
        
        # 3. 语言模块
        self.language_module = LanguageModule(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name
        )
        
        # 4. 多模态对齐
        self.alignment = ContrastiveAlignment(temperature=temperature)
        
        # 5. 变量编码器（用于生成H）
        self.variate_encoder = VariateEncoder(d_model=d_model)
        
        # 6. 变量选择模块
        self.variate_selection = VariateSelection(
            d_model=d_model,
            n_heads=n_heads
        )
        
        # 7. 生成器
        self.generator = Generator(
            d_model=d_model,
            n_patches=self.n_patches,
            prediction_steps=prediction_steps
        )
    
    def forward(self, x, return_loss=True):
        """
        前向传播
        
        输入:
            - x: [Batch, Time_Steps, N_Variates] 输入时间序列
            - return_loss: 是否返回对比损失
        
        输出:
            - Y_pred: [Batch, N_Variates, Prediction_Steps] 预测结果
            - contrastive_loss: 对比损失（如果return_loss=True）
        """
        # 1. 数据预处理
        # 视觉输入：[Batch, N_Variates, 3, H, W]
        images = self.visual_preprocessor(x)
        
        # 语言输入：[Batch, N_Variates, N_Patches, Patch_Length]
        patches = self.language_preprocessor(x)
        
        # 2. 视觉模块
        # CLS_img: [Batch, N_Variates, D_Model]
        CLS_img = self.vision_module(images)
        
        # 3. 语言模块
        # CLS_text: [Batch, N_Variates, D_Model]
        # Feat_text: [Batch, N_Variates, N_Patches + 1, D_Model]
        CLS_text, Feat_text = self.language_module(patches)
        
        # 4. 多模态对齐
        contrastive_loss = None
        if return_loss:
            contrastive_loss = self.alignment(CLS_img, CLS_text)
        
        # 5. 变量编码器
        # H: [Batch, N_Variates, D_Model]
        H = self.variate_encoder(x)
        
        # 6. 变量选择
        # v_CLS: [Batch, N_Variates, D_Model]
        v_CLS = self.variate_selection(CLS_text, H)
        
        # 7. 生成器
        # Y_pred: [Batch, N_Variates, Prediction_Steps]
        Y_pred = self.generator(Feat_text, v_CLS)
        
        if return_loss:
            return Y_pred, contrastive_loss
        else:
            return Y_pred
    
    def get_parameter_groups(self, lr_vision=1e-5, lr_other=1e-4):
        """
        获取不同学习率的参数组
        
        参数:
            - lr_vision: 预训练视觉模块投影层的学习率
            - lr_other: 其他部分的学习率
        
        返回:
            - parameter_groups: 参数组列表
        """
        # 视觉模块的投影层（需要较小学习率）
        vision_params = list(self.vision_module.projection.parameters())
        
        # 其他所有参数（语言模块、变量选择、生成器等）
        vision_param_ids = set(id(p) for p in vision_params)
        other_params = [p for p in self.parameters() if id(p) not in vision_param_ids and p.requires_grad]
        
        parameter_groups = [
            {'params': vision_params, 'lr': lr_vision},
            {'params': other_params, 'lr': lr_other}
        ]
        
        return parameter_groups
    
    def compute_loss(self, y_pred, y_true, contrastive_loss, lambda_gen=1.0, lambda_align=0.1):
        """
        计算总损失
        
        参数:
            - y_pred: [Batch, N_Variates, Prediction_Steps] 预测值
            - y_true: [Batch, N_Variates, Prediction_Steps] 真实值
            - contrastive_loss: 对比损失
            - lambda_gen: 生成损失权重
            - lambda_align: 对齐损失权重
        
        返回:
            - total_loss: 总损失
            - loss_dict: 各项损失的字典
        """
        # 生成损失（MSE）
        mse_loss = nn.MSELoss()(y_pred, y_true)
        
        # 总损失
        total_loss = lambda_gen * mse_loss + lambda_align * contrastive_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }
        
        return total_loss, loss_dict
    
    def encode_timeseries(self, x):
        """
        编码时间序列为特征向量（用于下游任务，如产量预测）
        
        输入:
            - x: [Batch, Time_Steps, N_Variates] 输入时间序列
        
        输出:
            - features: [Batch, D_Model] 时间序列特征向量
        """
        # 1. 语言输入：[Batch, N_Variates, N_Patches, Patch_Length]
        patches = self.language_preprocessor(x)
        
        # 2. 语言模块提取特征
        # CLS_text: [Batch, N_Variates, D_Model]
        CLS_text, _ = self.language_module(patches)
        
        # 3. 平均池化得到全局特征
        # features: [Batch, D_Model]
        features = CLS_text.mean(dim=1)
        
        return features

