"""
生成器模块
基于融合后的特征进行最终预测
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    生成器：将融合的多模态特征转换为最终预测
    输入:
        - Feat_text: [Batch, N_Variates, N_Patches + 1, D_Model]
        - v_CLS: [Batch, N_Variates, D_Model]
    输出:
        - Y_pred: [Batch, N_Variates, Prediction_Steps]
    """
    
    def __init__(self, d_model=512, n_patches=None, prediction_steps=96):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        self.prediction_steps = prediction_steps
        
        # 预测头：线性层
        # 输入维度：(N_Patches + 1) * D_Model（展平后的特征）
        # 输出维度：Prediction_Steps
        if n_patches is not None:
            input_dim = (n_patches + 1) * d_model
        else:
            # 如果n_patches未指定，在forward中动态计算
            input_dim = None
        
        if input_dim is not None:
            self.prediction_head = nn.Linear(input_dim, prediction_steps)
        else:
            self.prediction_head = None
    
    def _build_prediction_head(self, input_dim):
        """动态构建预测头"""
        self.prediction_head = nn.Linear(input_dim, self.prediction_steps)
        if next(self.parameters()).is_cuda:
            self.prediction_head = self.prediction_head.cuda()
    
    def forward(self, Feat_text, v_CLS):
        """
        前向传播
        输入:
            - Feat_text: [Batch, N_Variates, N_Patches + 1, D_Model]
            - v_CLS: [Batch, N_Variates, D_Model]
        输出:
            - Y_pred: [Batch, N_Variates, Prediction_Steps]
        """
        batch_size, n_variates, seq_len, d_model = Feat_text.shape
        
        # 特征融合：用v_CLS替换Feat_text序列的最后一个patch特征
        # 这样做是为了减少padding噪声的影响
        Feat_text_fused = Feat_text.clone()
        Feat_text_fused[:, :, -1, :] = v_CLS
        
        # 展平特征序列 [Batch, N_Variates, (N_Patches + 1) * D_Model]
        features_flat = Feat_text_fused.reshape(batch_size, n_variates, -1)
        
        # 动态构建预测头（如果还没有构建）
        if self.prediction_head is None:
            input_dim = features_flat.size(-1)
            self._build_prediction_head(input_dim)
        
        # 对每个变量分别预测
        predictions = []
        for v in range(n_variates):
            # 获取单个变量的特征 [Batch, Feature_Dim]
            feat_v = features_flat[:, v, :]
            
            # 通过预测头 [Batch, Prediction_Steps]
            pred_v = self.prediction_head(feat_v)
            
            predictions.append(pred_v)
        
        # 堆叠所有变量的预测 [Batch, N_Variates, Prediction_Steps]
        Y_pred = torch.stack(predictions, dim=1)
        
        return Y_pred

