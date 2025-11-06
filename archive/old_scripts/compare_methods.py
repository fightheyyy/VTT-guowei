"""
对比两种方法：
1. 两阶段法：先预测补全序列，再预测产量
2. 直接法：直接从少量序列预测产量
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ============================================================================
# 方法1：两阶段法（先补全序列，再预测产量）
# ============================================================================

class TwoStagePredictor(nn.Module):
    """
    两阶段预测器
    阶段1：前N个月 → 完整36个月
    阶段2：完整36个月 → 产量
    """
    def __init__(self, input_length=18, total_length=36, n_variates=7, d_model=128):
        super().__init__()
        self.input_length = input_length
        self.total_length = total_length
        self.n_variates = n_variates
        
        # 阶段1：序列补全
        self.stage1_encoder = nn.LSTM(
            input_size=n_variates,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True
        )
        self.stage1_decoder = nn.Linear(d_model, n_variates * (total_length - input_length))
        
        # 阶段2：产量预测
        self.stage2_encoder = nn.LSTM(
            input_size=n_variates,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True
        )
        self.stage2_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x, return_sequence=False):
        """
        x: [Batch, Input_Length, N_Variates]
        """
        batch_size = x.shape[0]
        
        # 阶段1：补全序列
        _, (h, _) = self.stage1_encoder(x)
        h = h[-1]  # 取最后一层
        
        pred_future = self.stage1_decoder(h)  # [Batch, (Total-Input)*N_Variates]
        pred_future = pred_future.view(batch_size, self.total_length - self.input_length, self.n_variates)
        
        # 拼接完整序列
        full_sequence = torch.cat([x, pred_future], dim=1)  # [Batch, Total_Length, N_Variates]
        
        # 阶段2：产量预测
        _, (h2, _) = self.stage2_encoder(full_sequence)
        h2 = h2[-1]
        
        yield_pred = self.stage2_predictor(h2)  # [Batch, 1]
        
        if return_sequence:
            return yield_pred, full_sequence
        return yield_pred


# ============================================================================
# 方法2：直接法（直接从少量序列预测产量）
# ============================================================================

class DirectPredictor(nn.Module):
    """
    直接预测器
    前N个月 → 产量（端到端）
    """
    def __init__(self, input_length=18, n_variates=7, d_model=128):
        super().__init__()
        
        self.encoder = nn.LSTM(
            input_size=n_variates,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        """
        x: [Batch, Input_Length, N_Variates]
        """
        # 编码
        _, (h, _) = self.encoder(x)
        h = h[-1]  # [Batch, D_Model]
        
        # 预测产量
        yield_pred = self.predictor(h)  # [Batch, 1]
        
        return yield_pred


# ============================================================================
# 数据加载
# ============================================================================

class YieldDataset(Dataset):
    """产量预测数据集"""
    
    def __init__(self, csv_paths, selected_bands, input_length=18, total_length=36):
        self.selected_bands = selected_bands
        self.input_length = input_length
        self.total_length = total_length
        
        # 加载数据
        all_sequences = []
        all_yields = []
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            
            # 提取序列
            sequences = []
            for band in selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(total_length)]
                band_data = df[cols].values
                sequences.append(band_data)
            
            sequences = np.stack(sequences, axis=2)  # [N, Time, Vars]
            all_sequences.append(sequences)
            
            # 提取产量（使用2022年的产量）
            yields = df['y2022'].values.reshape(-1, 1)
            all_yields.append(yields)
        
        self.sequences = np.concatenate(all_sequences, axis=0)
        self.yields = np.concatenate(all_yields, axis=0)
        
        # 归一化
        self.seq_mean = self.sequences.mean(axis=(0, 1), keepdims=True)
        self.seq_std = self.sequences.std(axis=(0, 1), keepdims=True)
        self.seq_std = np.where(self.seq_std < 1e-6, 1.0, self.seq_std)
        self.sequences = (self.sequences - self.seq_mean) / self.seq_std
        
        self.yield_mean = self.yields.mean()
        self.yield_std = self.yields.std()
        self.yields = (self.yields - self.yield_mean) / self.yield_std
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # 输入：前N个月
        x_input = self.sequences[idx, :self.input_length, :]
        
        # 目标1：后面的月份（用于两阶段训练）
        x_future = self.sequences[idx, self.input_length:, :]
        
        # 目标2：产量
        y_yield = self.yields[idx]
        
        return (
            torch.FloatTensor(x_input),
            torch.FloatTensor(x_future),
            torch.FloatTensor(y_yield)
        )


def create_yield_dataloaders(train_files, test_files, selected_bands, input_length=18, batch_size=16):
    train_dataset = YieldDataset(train_files, selected_bands, input_length=input_length)
    test_dataset = YieldDataset(test_files, selected_bands, input_length=input_length)
    
    # 使用训练集的归一化参数
    test_dataset.seq_mean = train_dataset.seq_mean
    test_dataset.seq_std = train_dataset.seq_std
    test_dataset.yield_mean = train_dataset.yield_mean
    test_dataset.yield_std = train_dataset.yield_std
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.yield_mean, train_dataset.yield_std


# ============================================================================
# 训练和评估
# ============================================================================

def train_two_stage(model, train_loader, test_loader, epochs=50, device='cuda'):
    """训练两阶段模型"""
    print("\n" + "="*70)
    print("训练两阶段模型")
    print("="*70)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    seq_criterion = nn.MSELoss()
    yield_criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_input, x_future, y_yield in train_loader:
            x_input = x_input.to(device)
            x_future = x_future.to(device)
            y_yield = y_yield.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            yield_pred, full_seq = model(x_input, return_sequence=True)
            pred_future = full_seq[:, model.input_length:, :]
            
            # 两个损失：序列重建 + 产量预测
            loss_seq = seq_criterion(pred_future, x_future)
            loss_yield = yield_criterion(yield_pred, y_yield)
            
            # 总损失（序列重建权重较小）
            loss = 0.3 * loss_seq + 1.0 * loss_yield
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_input, x_future, y_yield in test_loader:
                x_input = x_input.to(device)
                y_yield = y_yield.to(device)
                
                yield_pred = model(x_input)
                loss = yield_criterion(yield_pred, y_yield)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")
    
    model.load_state_dict(best_model_state)
    return model


def train_direct(model, train_loader, test_loader, epochs=50, device='cuda'):
    """训练直接模型"""
    print("\n" + "="*70)
    print("训练直接模型")
    print("="*70)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_input, _, y_yield in train_loader:
            x_input = x_input.to(device)
            y_yield = y_yield.to(device)
            
            optimizer.zero_grad()
            
            yield_pred = model(x_input)
            loss = criterion(yield_pred, y_yield)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_input, _, y_yield in test_loader:
                x_input = x_input.to(device)
                y_yield = y_yield.to(device)
                
                yield_pred = model(x_input)
                loss = criterion(yield_pred, y_yield)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")
    
    model.load_state_dict(best_model_state)
    return model


def evaluate(model, test_loader, yield_mean, yield_std, device='cuda'):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for x_input, _, y_yield in test_loader:
            x_input = x_input.to(device)
            
            yield_pred = model(x_input)
            
            all_preds.extend(yield_pred.cpu().numpy())
            all_trues.extend(y_yield.numpy())
    
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
    # 反归一化
    all_preds = all_preds * yield_std + yield_mean
    all_trues = all_trues * yield_std + yield_mean
    
    # 计算指标
    mse = mean_squared_error(all_trues, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_trues, all_preds)
    r2 = r2_score(all_trues, all_preds)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# ============================================================================
# 主实验
# ============================================================================

def main():
    print("="*70)
    print("对比实验：两阶段法 vs 直接法")
    print("="*70)
    
    # 配置
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = ["extract2022_20251010_165007.csv"]
    
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试不同的输入长度
    input_lengths = [6, 12, 18, 24]
    
    results = {}
    
    for input_len in input_lengths:
        print(f"\n{'='*70}")
        print(f"实验：使用前{input_len}个月预测产量")
        print(f"{'='*70}")
        
        # 加载数据
        train_loader, test_loader, yield_mean, yield_std = create_yield_dataloaders(
            train_files, test_files, selected_bands, 
            input_length=input_len, batch_size=32
        )
        
        # 方法1：两阶段
        model_two_stage = TwoStagePredictor(
            input_length=input_len, 
            total_length=36, 
            n_variates=7,
            d_model=128
        ).to(device)
        
        model_two_stage = train_two_stage(
            model_two_stage, train_loader, test_loader, 
            epochs=50, device=device
        )
        
        results_two_stage = evaluate(
            model_two_stage, test_loader, yield_mean, yield_std, device
        )
        
        # 方法2：直接法
        model_direct = DirectPredictor(
            input_length=input_len, 
            n_variates=7,
            d_model=128
        ).to(device)
        
        model_direct = train_direct(
            model_direct, train_loader, test_loader, 
            epochs=50, device=device
        )
        
        results_direct = evaluate(
            model_direct, test_loader, yield_mean, yield_std, device
        )
        
        # 保存结果
        results[input_len] = {
            'two_stage': results_two_stage,
            'direct': results_direct
        }
        
        # 打印对比
        print(f"\n前{input_len}个月预测产量结果:")
        print(f"{'─'*70}")
        print(f"{'方法':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
        print(f"{'─'*70}")
        print(f"{'两阶段法':<20} {results_two_stage['rmse']:>10.4f} {results_two_stage['mae']:>10.4f} {results_two_stage['r2']:>10.4f}")
        print(f"{'直接法':<20} {results_direct['rmse']:>10.4f} {results_direct['mae']:>10.4f} {results_direct['r2']:>10.4f}")
        
        # 计算提升
        improvement = (results_two_stage['rmse'] - results_direct['rmse']) / results_two_stage['rmse'] * 100
        print(f"\n直接法相比两阶段法: {improvement:+.1f}% RMSE")
        
        if improvement > 0:
            print("✅ 直接法更好！")
        else:
            print("❌ 两阶段法更好")
    
    # 总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    
    print(f"\n{'输入长度':<15} {'两阶段RMSE':>15} {'直接法RMSE':>15} {'直接法提升':>15}")
    print("─"*70)
    for input_len in input_lengths:
        two_stage_rmse = results[input_len]['two_stage']['rmse']
        direct_rmse = results[input_len]['direct']['rmse']
        improvement = (two_stage_rmse - direct_rmse) / two_stage_rmse * 100
        
        print(f"{input_len:>3}个月{'':<10} {two_stage_rmse:>15.4f} {direct_rmse:>15.4f} {improvement:>14.1f}%")
    
    print("\n结论:")
    avg_improvement = np.mean([
        (results[l]['two_stage']['rmse'] - results[l]['direct']['rmse']) / results[l]['two_stage']['rmse'] * 100
        for l in input_lengths
    ])
    
    if avg_improvement > 5:
        print(f"✅ 直接法平均提升 {avg_improvement:.1f}%，明显优于两阶段法")
        print("   原因：避免了序列预测误差的累积，端到端优化更有效")
    elif avg_improvement < -5:
        print(f"❌ 两阶段法平均提升 {-avg_improvement:.1f}%，明显优于直接法")
        print("   原因：完整序列提供了更多信息，补偿了预测误差")
    else:
        print(f"⚖ 两种方法性能接近（差异 {abs(avg_improvement):.1f}%）")
        print("   建议：使用直接法（更简单高效）")


if __name__ == "__main__":
    main()

