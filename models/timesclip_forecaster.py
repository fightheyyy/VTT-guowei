"""
TimesCLIP序列预测器
基于TimesCLIP的双模态编码器进行时间序列预测
输入: 部分序列 (如前6步)
输出: 完整序列 (37步)
"""

import torch
import torch.nn as nn
from models.timesclip_classifier import TimesCLIPClassifier


class TimesCLIPForecaster(nn.Module):
    """
    基于TimesCLIP的序列预测器
    
    架构:
    1. 使用TimesCLIP编码器提取早期序列特征
    2. 通过解码器预测剩余序列
    3. 拼接得到完整序列
    """
    
    def __init__(
        self,
        input_len=6,           # 输入序列长度
        output_len=37,         # 完整序列长度
        n_variates=14,         # 变量数
        d_model=512,
        patch_length=2,        # 减小patch_length以适应短序列
        stride=1,              # 使用滑动窗口
        clip_model_name="openai/clip-vit-base-patch16",
        use_vision=True,       # 是否使用视觉分支
        use_language=True,     # 是否使用语言分支
        decoder_type='mlp',    # 'mlp', 'lstm', 'transformer'
        hidden_dim=512,
        num_decoder_layers=3,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.pred_len = output_len - input_len
        self.n_variates = n_variates
        self.use_vision = use_vision
        self.use_language = use_language
        self.decoder_type = decoder_type
        
        # ============ 编码器：TimesCLIP ============
        # 使用TimesCLIP的编码部分
        self.timesclip = TimesCLIPClassifier(
            time_steps=input_len,
            n_variates=n_variates,
            num_classes=4,  # dummy，不使用分类头
            d_model=d_model,
            patch_length=patch_length,
            stride=stride,
            clip_model_name=clip_model_name,
            use_variate_selection=False,  # 预测任务不需要变量选择
            use_contrastive=False,  # 预测任务不需要对比学习
            dropout=dropout
        )
        
        # 计算编码特征维度
        feature_dim = 0
        if use_vision:
            feature_dim += d_model
        if use_language:
            feature_dim += d_model
        # 预测任务不使用variate selection，特征维度更简洁
        
        # ============ 解码器 ============
        if decoder_type == 'mlp':
            self.decoder = MLPDecoder(
                feature_dim=feature_dim,
                output_len=self.pred_len,
                n_variates=n_variates,
                hidden_dim=hidden_dim,
                num_layers=num_decoder_layers,
                dropout=dropout
            )
        elif decoder_type == 'lstm':
            self.decoder = LSTMDecoder(
                feature_dim=feature_dim,
                output_len=self.pred_len,
                n_variates=n_variates,
                hidden_dim=hidden_dim,
                num_layers=num_decoder_layers,
                dropout=dropout
            )
        elif decoder_type == 'transformer':
            self.decoder = TransformerDecoder(
                feature_dim=feature_dim,
                output_len=self.pred_len,
                n_variates=n_variates,
                d_model=hidden_dim,
                num_layers=num_decoder_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def encode(self, x, cached_images=None):
        """
        使用TimesCLIP编码器提取特征
        
        Args:
            x: [batch_size, input_len, n_variates]
            cached_images: 可选的预生成图像
        
        Returns:
            features: [batch_size, feature_dim]
        """
        batch_size = x.shape[0]
        
        # 提取双模态特征
        features_list = []
        
        # 语言分支特征
        if self.use_language:
            # 使用语言预处理器和模块
            patches = self.timesclip.language_preprocessor(x)
            lang_output = self.timesclip.language_module(patches)
            # language_module返回(features, CLS_token)，我们只需要features
            if isinstance(lang_output, tuple):
                lang_features = lang_output[0]  # 取第一个元素
            else:
                lang_features = lang_output
            features_list.append(lang_features)
        
        # 视觉分支特征
        if self.use_vision:
            if cached_images is not None:
                # 使用缓存图像
                images = cached_images
            else:
                # 动态生成图像
                images = []
                for i in range(batch_size):
                    img = self.timesclip.visual_preprocessor.time_series_to_image(
                        x[i].cpu().numpy()
                    )
                    images.append(img)
                images = torch.stack(images).to(x.device)
            
            vis_features = self.timesclip.vision_module(images)
            features_list.append(vis_features)
        
        # 拼接所有特征
        features = torch.cat(features_list, dim=1)
        
        return features
    
    def forward(self, x, cached_images=None, return_encoding=False):
        """
        前向传播
        
        Args:
            x: [batch_size, input_len, n_variates] 输入序列
            cached_images: 可选的预生成图像
            return_encoding: 是否返回编码特征
        
        Returns:
            x_full: [batch_size, output_len, n_variates] 完整序列
            或 (x_full, features) 如果return_encoding=True
        """
        # 编码
        features = self.encode(x, cached_images)
        
        # 解码：预测剩余序列
        x_pred = self.decoder(features, x)  # [batch_size, pred_len, n_variates]
        
        # 拼接：[输入序列 + 预测序列]
        x_full = torch.cat([x, x_pred], dim=1)  # [batch_size, output_len, n_variates]
        
        if return_encoding:
            return x_full, features
        return x_full


class MLPDecoder(nn.Module):
    """MLP解码器 - 简单直接"""
    
    def __init__(self, feature_dim, output_len, n_variates, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.output_len = output_len
        self.n_variates = n_variates
        
        # 构建MLP层
        layers = []
        in_dim = feature_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, output_len * n_variates))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, features, x_input):
        """
        Args:
            features: [batch_size, feature_dim] 或 [batch_size, seq, feature_dim]
            x_input: [batch_size, input_len, n_variates] (用于残差连接)
        
        Returns:
            x_pred: [batch_size, output_len, n_variates]
        """
        batch_size = features.shape[0]
        
        # 处理特征维度 - 确保是2D [batch, feature_dim]
        if features.dim() > 2:
            # 如果是3D或4D，使用全局平均池化降维
            while features.dim() > 2:
                features = features.mean(dim=1)  # [batch, ...]
        
        # MLP预测
        out = self.net(features)  # [batch_size, output_len * n_variates]
        
        # reshape
        x_pred = out.view(batch_size, self.output_len, self.n_variates)
        
        return x_pred


class LSTMDecoder(nn.Module):
    """LSTM解码器 - 考虑时间依赖"""
    
    def __init__(self, feature_dim, output_len, n_variates, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.output_len = output_len
        self.n_variates = n_variates
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 特征投影
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_variates,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, n_variates)
    
    def forward(self, features, x_input):
        """
        Args:
            features: [batch_size, feature_dim] 或 [batch_size, seq, feature_dim]
            x_input: [batch_size, input_len, n_variates]
        
        Returns:
            x_pred: [batch_size, output_len, n_variates]
        """
        batch_size = features.shape[0]
        
        # 处理特征维度 - 确保是2D [batch, feature_dim]
        if features.dim() > 2:
            # 如果是3D或4D，使用全局平均池化降维
            while features.dim() > 2:
                features = features.mean(dim=1)  # [batch, ...]
        
        # 初始化隐藏状态
        h0 = self.feature_proj(features)  # [batch_size, hidden_dim]
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, hidden_dim]
        c0 = torch.zeros_like(h0)
        
        # 自回归预测
        predictions = []
        current_input = x_input[:, -1:, :]  # 从最后一个时间步开始
        
        for _ in range(self.output_len):
            # LSTM步进
            out, (h0, c0) = self.lstm(current_input, (h0, c0))
            
            # 预测下一个时间步
            pred = self.output_proj(out)  # [batch_size, 1, n_variates]
            predictions.append(pred)
            
            # 更新输入
            current_input = pred
        
        # 拼接所有预测
        x_pred = torch.cat(predictions, dim=1)  # [batch_size, output_len, n_variates]
        
        return x_pred


class TransformerDecoder(nn.Module):
    """Transformer解码器 - 并行预测"""
    
    def __init__(self, feature_dim, output_len, n_variates, d_model=512, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.output_len = output_len
        self.n_variates = n_variates
        self.d_model = d_model
        
        # 特征投影
        self.feature_proj = nn.Linear(feature_dim, d_model)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, output_len, d_model))
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, n_variates)
    
    def forward(self, features, x_input):
        """
        Args:
            features: [batch_size, feature_dim] 或 [batch_size, seq, feature_dim]
            x_input: [batch_size, input_len, n_variates]
        
        Returns:
            x_pred: [batch_size, output_len, n_variates]
        """
        batch_size = features.shape[0]
        
        # 处理特征维度 - 确保是2D [batch, feature_dim]
        if features.dim() > 2:
            # 如果是3D或4D，使用全局平均池化降维
            while features.dim() > 2:
                features = features.mean(dim=1)  # [batch, ...]
        
        # 特征作为memory
        memory = self.feature_proj(features).unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 查询（位置编码）
        queries = self.pos_embedding.repeat(batch_size, 1, 1)  # [batch_size, output_len, d_model]
        
        # Transformer解码
        out = self.transformer_decoder(queries, memory)  # [batch_size, output_len, d_model]
        
        # 输出投影
        x_pred = self.output_proj(out)  # [batch_size, output_len, n_variates]
        
        return x_pred


if __name__ == "__main__":
    # 测试代码
    print("测试TimesCLIPForecaster...")
    
    # 创建模型
    model = TimesCLIPForecaster(
        input_len=6,
        output_len=37,
        n_variates=14,
        decoder_type='mlp',
        use_vision=True,
        use_language=True
    )
    
    # 测试数据
    x = torch.randn(4, 6, 14)  # batch=4, 前6步
    
    # 前向传播
    x_full = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {x_full.shape}")
    assert x_full.shape == (4, 37, 14), "输出形状错误"
    
    print("\n测试不同解码器...")
    for decoder_type in ['mlp', 'lstm', 'transformer']:
        model = TimesCLIPForecaster(
            input_len=6,
            output_len=37,
            n_variates=14,
            decoder_type=decoder_type
        )
        x_full = model(x)
        print(f"{decoder_type}: {x_full.shape} ✓")
    
    print("\n所有测试通过!")

