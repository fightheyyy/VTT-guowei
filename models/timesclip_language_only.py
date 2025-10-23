"""
TimesCLIP 简化版：只使用语言模态
用于消融实验，对比视觉模态的作用
"""

import torch
import torch.nn as nn
from .preprocessor import LanguagePreprocessor
from .language_module import LanguageModule
from .variate_selection import VariateSelection, VariateEncoder
from .generator import Generator


class TimesCLIPLanguageOnly(nn.Module):
    """
    TimesCLIP 语言模态版本
    去掉了视觉预处理器、视觉模块和对比学习对齐
    
    参数:
        - time_steps: 输入时间序列长度
        - n_variates: 变量数量
        - prediction_steps: 预测步长
        - patch_length: patch长度
        - stride: patch步长
        - d_model: 模型隐藏维度
        - n_heads: 注意力头数
        - clip_model_name: CLIP模型名称（用于LanguageModule）
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
        
        # 1. 语言预处理器
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        
        # 2. 语言模块
        self.language_module = LanguageModule(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name
        )
        
        # 3. 变量编码器（用于生成H）
        self.variate_encoder = VariateEncoder(d_model=d_model)
        
        # 4. 变量选择模块
        self.variate_selection = VariateSelection(
            d_model=d_model,
            n_heads=n_heads
        )
        
        # 5. 生成器
        self.generator = Generator(
            d_model=d_model,
            n_patches=self.n_patches,
            prediction_steps=prediction_steps
        )
    
    def forward(self, x):
        """
        前向传播（无对比损失）
        
        输入:
            - x: [Batch, Time_Steps, N_Variates] 输入时间序列
        
        输出:
            - Y_pred: [Batch, N_Variates, Prediction_Steps] 预测结果
        """
        # 1. 语言预处理
        # 语言输入：[Batch, N_Variates, N_Patches, Patch_Length]
        patches = self.language_preprocessor(x)
        
        # 2. 语言模块
        # CLS_text: [Batch, N_Variates, D_Model]
        # Feat_text: [Batch, N_Variates, N_Patches + 1, D_Model]
        CLS_text, Feat_text = self.language_module(patches)
        
        # 3. 变量编码器
        # H: [Batch, N_Variates, D_Model]
        H = self.variate_encoder(x)
        
        # 4. 变量选择
        # v_CLS: [Batch, N_Variates, D_Model]
        v_CLS = self.variate_selection(CLS_text, H)
        
        # 5. 生成器
        # Y_pred: [Batch, N_Variates, Prediction_Steps]
        Y_pred = self.generator(Feat_text, v_CLS)
        
        return Y_pred
    
    def get_parameter_groups(self, lr_language=1e-5, lr_other=1e-4):
        """
        获取不同学习率的参数组
        
        参数:
            - lr_language: BERT投影层的学习率
            - lr_other: 其他部分的学习率
        
        返回:
            - parameter_groups: 参数组列表
        """
        # 语言模块的参数（需要较小学习率）
        language_params = list(self.language_module.parameters())
        
        # 其他所有参数
        language_param_ids = set(id(p) for p in language_params)
        other_params = [p for p in self.parameters() if id(p) not in language_param_ids and p.requires_grad]
        
        parameter_groups = [
            {'params': language_params, 'lr': lr_language},
            {'params': other_params, 'lr': lr_other}
        ]
        
        return parameter_groups
    
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


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    time_steps = 18
    n_variates = 7
    prediction_steps = 18
    
    model = TimesCLIPLanguageOnly(
        time_steps=time_steps,
        n_variates=n_variates,
        prediction_steps=prediction_steps,
        patch_length=6,
        stride=3,
        d_model=256
    )
    
    x = torch.randn(batch_size, time_steps, n_variates)
    y_pred = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_pred.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

