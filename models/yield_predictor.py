"""
产量预测模型
基于完整的全年波段时间序列数据预测产量
"""

import torch
import torch.nn as nn


class YieldPredictor(nn.Module):
    """
    产量预测器
    输入: [Batch, 36, N_Variates] 全年波段时间序列
    输出: [Batch, 1] 产量值
    """
    
    def __init__(self, n_variates=7, time_steps=36, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_variates = n_variates
        self.time_steps = time_steps
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(n_variates, d_model)
        
        # 位置编码
        self.position_embeddings = nn.Parameter(torch.randn(1, time_steps, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 全局池化
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        前向传播
        输入: x [Batch, Time_Steps, N_Variates]
        输出: yield_pred [Batch, 1]
        """
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # [Batch, Time_Steps, D_Model]
        
        # 添加位置编码
        x = x + self.position_embeddings
        
        # LayerNorm
        x = self.layer_norm(x)
        
        # Transformer编码
        x = self.transformer(x)  # [Batch, Time_Steps, D_Model]
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [Batch, D_Model, Time_Steps]
        x = self.pooling(x).squeeze(-1)  # [Batch, D_Model]
        
        # 预测产量
        yield_pred = self.predictor(x)  # [Batch, 1]
        
        return yield_pred


class CombinedYieldPredictor(nn.Module):
    """
    组合模型：集成多个特征提取器
    """
    
    def __init__(self, n_variates=7, time_steps=36, d_model=256):
        super().__init__()
        
        # 时序特征提取器
        self.temporal_encoder = YieldPredictor(
            n_variates=n_variates,
            time_steps=time_steps,
            d_model=d_model,
            n_heads=8,
            n_layers=4
        )
        
        # 直接从时间序列提取统计特征
        self.stat_extractor = nn.Sequential(
            nn.Linear(n_variates * time_steps, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        """
        输入: x [Batch, Time_Steps, N_Variates]
        输出: yield_pred [Batch, 1]
        """
        batch_size = x.size(0)
        
        # 时序特征
        temporal_feat = self.temporal_encoder.transformer(
            self.temporal_encoder.layer_norm(
                self.temporal_encoder.input_projection(x) + 
                self.temporal_encoder.position_embeddings
            )
        )
        temporal_feat = temporal_feat.mean(dim=1)  # [Batch, D_Model]
        
        # 统计特征
        stat_feat = self.stat_extractor(x.reshape(batch_size, -1))  # [Batch, D_Model//2]
        
        # 融合预测
        combined = torch.cat([temporal_feat, stat_feat], dim=1)
        yield_pred = self.fusion(combined)
        
        return yield_pred


if __name__ == "__main__":
    # 测试模型
    batch_size = 8
    n_variates = 7
    time_steps = 36
    
    model = YieldPredictor(n_variates=n_variates, time_steps=time_steps, d_model=256)
    
    x = torch.randn(batch_size, time_steps, n_variates)
    y_pred = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_pred.shape}")
    print(f"产量预测值: {y_pred.squeeze().detach().numpy()}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

