"""
TimesCLIP 可变长度版本：支持任意前N个月预测剩余月份
实现早期预测功能
"""

import torch
import torch.nn as nn
from .preprocessor import LanguagePreprocessor
from .language_module import LanguageModule
from .variate_selection import VariateSelection, VariateEncoder
from .generator import Generator


class TimesCLIPVariableLength(nn.Module):
    """
    TimesCLIP 可变长度版本
    支持输入任意长度的前N个月，预测剩余月份
    
    参数:
        - max_time_steps: 最大时间步数（如36个月）
        - n_variates: 变量数量
        - patch_length: patch长度
        - stride: patch步长
        - d_model: 模型隐藏维度
        - n_heads: 注意力头数
        - clip_model_name: CLIP模型名称
    """
    
    def __init__(
        self,
        max_time_steps=36,
        n_variates=7,
        patch_length=6,
        stride=3,
        d_model=512,
        n_heads=8,
        clip_model_name="openai/clip-vit-base-patch16"
    ):
        super().__init__()
        
        self.max_time_steps = max_time_steps
        self.n_variates = n_variates
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        
        # 语言预处理器（动态处理不同长度）
        self.language_preprocessor = LanguagePreprocessor(
            patch_length=patch_length,
            stride=stride
        )
        
        # 语言模块
        self.language_module = LanguageModule(
            patch_length=patch_length,
            d_model=d_model,
            clip_model_name=clip_model_name
        )
        
        # 变量编码器
        self.variate_encoder = VariateEncoder(d_model=d_model)
        
        # 变量选择模块
        self.variate_selection = VariateSelection(
            d_model=d_model,
            n_heads=n_heads
        )
        
        # 位置编码（用于不同输入长度）
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_time_steps, d_model)
        )
        
        # 长度嵌入（告诉模型当前输入长度）
        self.length_embedding = nn.Embedding(max_time_steps, d_model)
        
        # 解码器：基于已知序列生成未来序列
        # 使用 Transformer Decoder 架构
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, n_variates)
        
    def forward(self, x, input_length=None):
        """
        前向传播
        
        输入:
            - x: [Batch, Input_Length, N_Variates] 前N个月的数据
            - input_length: 输入长度（如果为None，则从x.shape[1]推断）
        
        输出:
            - Y_pred: [Batch, Prediction_Length, N_Variates] 预测剩余月份
        """
        batch_size, input_len, n_vars = x.shape
        
        if input_length is None:
            input_length = input_len
        
        # 计算需要预测的长度
        prediction_len = self.max_time_steps - input_length
        
        # 1. 编码输入序列
        # patches: [Batch, N_Variates, N_Patches, Patch_Length]
        patches = self.language_preprocessor(x)
        
        # CLS_text: [Batch, N_Variates, D_Model]
        # Feat_text: [Batch, N_Variates, N_Patches + 1, D_Model]
        CLS_text, Feat_text = self.language_module(patches)
        
        # 变量编码
        H = self.variate_encoder(x)
        
        # 变量选择
        v_CLS = self.variate_selection(CLS_text, H)
        
        # 2. 融合特征作为记忆（memory）
        # memory: [Batch, N_Variates, D_Model]
        memory = CLS_text + v_CLS
        
        # 扩展为序列形式: [Batch, Input_Length, D_Model]
        # 使用变量的平均特征
        memory_seq = memory.mean(dim=1, keepdim=True).expand(batch_size, input_len, self.d_model)
        
        # 添加位置编码
        memory_seq = memory_seq + self.positional_encoding[:, :input_len, :]
        
        # 3. 生成未来序列
        # 初始化目标序列（用0填充）
        tgt = torch.zeros(batch_size, prediction_len, self.d_model, device=x.device)
        
        # 添加位置编码
        tgt = tgt + self.positional_encoding[:, input_len:input_len + prediction_len, :]
        
        # 添加长度信息
        length_emb = self.length_embedding(
            torch.tensor([input_length - 1], device=x.device)
        ).unsqueeze(0).expand(batch_size, prediction_len, -1)
        tgt = tgt + length_emb
        
        # Decoder生成
        # tgt: [Batch, Prediction_Length, D_Model]
        decoded = self.decoder(tgt, memory_seq)
        
        # 投影到变量空间
        # Y_pred: [Batch, Prediction_Length, N_Variates]
        Y_pred = self.output_projection(decoded)
        
        return Y_pred
    
    def predict_full_sequence(self, x):
        """
        预测完整序列（包含输入部分）
        
        输入:
            - x: [Batch, Input_Length, N_Variates] 前N个月
        
        输出:
            - full_seq: [Batch, Max_Time_Steps, N_Variates] 完整36个月序列
        """
        input_len = x.shape[1]
        
        # 预测未来部分
        future_pred = self.forward(x)
        
        # 拼接已知和预测部分
        full_seq = torch.cat([x, future_pred], dim=1)
        
        return full_seq
    
    def encode_timeseries(self, x):
        """
        编码时间序列为特征向量（用于产量预测）
        
        输入:
            - x: [Batch, Input_Length, N_Variates]
        
        输出:
            - features: [Batch, D_Model]
        """
        patches = self.language_preprocessor(x)
        CLS_text, _ = self.language_module(patches)
        features = CLS_text.mean(dim=1)
        return features


class DynamicLanguagePreprocessor(nn.Module):
    """
    支持可变长度输入的预处理器
    """
    def __init__(self, patch_length=6, stride=3):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
    
    def forward(self, x):
        """
        输入: [Batch, Time_Steps, N_Variates]
        输出: [Batch, N_Variates, N_Patches, Patch_Length]
        """
        batch_size, time_steps, n_variates = x.shape
        
        # 计算patch数量（动态）
        n_patches = (time_steps - self.patch_length) // self.stride + 1
        
        # 转置为 [Batch, N_Variates, Time_Steps]
        x = x.transpose(1, 2)
        
        # 提取patches
        patches = []
        for i in range(n_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_length
            patch = x[:, :, start_idx:end_idx]
            patches.append(patch)
        
        # 堆叠: [Batch, N_Variates, N_Patches, Patch_Length]
        patches = torch.stack(patches, dim=2)
        
        return patches


if __name__ == "__main__":
    print("=" * 70)
    print("测试可变长度预测模型")
    print("=" * 70)
    
    batch_size = 4
    max_time_steps = 36
    n_variates = 7
    
    model = TimesCLIPVariableLength(
        max_time_steps=max_time_steps,
        n_variates=n_variates,
        patch_length=6,
        stride=3,
        d_model=256
    )
    
    # 测试不同输入长度
    test_lengths = [3, 6, 12, 18, 24]
    
    for input_len in test_lengths:
        print(f"\n{'='*70}")
        print(f"测试：前{input_len}个月 → 预测后{max_time_steps - input_len}个月")
        print(f"{'='*70}")
        
        # 创建测试数据
        x = torch.randn(batch_size, input_len, n_variates)
        
        # 预测
        y_pred = model.forward(x)
        
        print(f"输入形状: {x.shape}")
        print(f"预测形状: {y_pred.shape}")
        print(f"预测长度: {y_pred.shape[1]} 个月")
        
        # 预测完整序列
        full_seq = model.predict_full_sequence(x)
        print(f"完整序列: {full_seq.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"模型统计")
    print(f"{'='*70}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

