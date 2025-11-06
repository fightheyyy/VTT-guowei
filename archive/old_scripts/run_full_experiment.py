"""
完整实验框架
实验A: 补全 vs 不补全
实验B: 不同补全长度
实验C: 消融实验
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# 模型定义
# ============================================================================

class DirectRegressor(nn.Module):
    """直接回归模型（基线）"""
    def __init__(self, input_length, n_variates=7, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_variates,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return self.regressor(h[-1])


class TwoStageRegressor(nn.Module):
    """两阶段模型：补全 + 回归"""
    def __init__(self, input_length, target_length, n_variates=7, hidden_dim=128):
        super().__init__()
        self.input_length = input_length
        self.target_length = target_length
        self.n_variates = n_variates
        
        # Stage1: 序列补全
        self.completion_encoder = nn.LSTM(
            input_size=n_variates,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.completion_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (target_length - input_length) * n_variates)
        )
        
        # Stage2: 回归
        self.regression_encoder = nn.LSTM(
            input_size=n_variates,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        self.stage1_trained = False
        self.stage2_trained = False
    
    def complete_sequence(self, x):
        """Stage1: 补全序列"""
        batch_size = x.shape[0]
        _, (h, _) = self.completion_encoder(x)
        pred_future = self.completion_decoder(h[-1])
        pred_future = pred_future.view(batch_size, self.target_length - self.input_length, self.n_variates)
        full_sequence = torch.cat([x, pred_future], dim=1)
        return full_sequence, pred_future
    
    def forward(self, x, return_completion=False):
        """完整前向传播"""
        # 补全序列
        full_sequence, pred_future = self.complete_sequence(x)
        
        # 回归预测
        _, (h, _) = self.regression_encoder(full_sequence)
        yield_pred = self.regressor(h[-1])
        
        if return_completion:
            return yield_pred, full_sequence, pred_future
        return yield_pred


# ============================================================================
# 数据集
# ============================================================================

class FlexibleYieldDataset(Dataset):
    """灵活的产量预测数据集"""
    def __init__(self, csv_paths, selected_bands, input_length, target_length=36):
        self.selected_bands = selected_bands
        self.input_length = input_length
        self.target_length = target_length
        
        # 加载数据
        all_sequences = []
        all_yields = []
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            
            # 提取序列
            sequences = []
            for band in selected_bands:
                cols = [f"{band}_{i:02d}" for i in range(target_length)]
                band_data = df[cols].values
                sequences.append(band_data)
            
            sequences = np.stack(sequences, axis=2)
            all_sequences.append(sequences)
            
            # 提取产量
            yields = df['y2022'].values.reshape(-1, 1)
            all_yields.append(yields)
        
        self.sequences = np.concatenate(all_sequences, axis=0)
        self.yields = np.concatenate(all_yields, axis=0)
        
        # 归一化
        self.seq_mean = self.sequences.mean(axis=(0, 1), keepdims=True)
        self.seq_std = self.sequences.std(axis=(0, 1), keepdims=True) + 1e-6
        self.sequences = (self.sequences - self.seq_mean) / self.seq_std
        
        self.yield_mean = self.yields.mean()
        self.yield_std = self.yields.std() + 1e-6
        self.yields = (self.yields - self.yield_mean) / self.yield_std
        
        print(f"  数据集大小: {len(self.sequences)}")
        print(f"  输入长度: {input_length}, 目标长度: {target_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x_input = self.sequences[idx, :self.input_length, :]
        x_full = self.sequences[idx, :self.target_length, :]
        y_yield = self.yields[idx]
        
        return (
            torch.FloatTensor(x_input),
            torch.FloatTensor(x_full),
            torch.FloatTensor(y_yield)
        )


# ============================================================================
# 训练函数
# ============================================================================

def train_direct_model(model, train_loader, test_loader, epochs=50, lr=1e-3, device='cuda'):
    """训练直接回归模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x_input, _, y_yield in train_loader:
            x_input = x_input.to(device)
            y_yield = y_yield.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_input)
            loss = criterion(y_pred, y_yield)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_input, _, y_yield in test_loader:
                x_input = x_input.to(device)
                y_yield = y_yield.to(device)
                y_pred = model(x_input)
                loss = criterion(y_pred, y_yield)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")
    
    model.load_state_dict(best_state)
    return model, history


def train_twostage_model(model, train_loader, test_loader, epochs=50, lr=1e-3, device='cuda'):
    """训练两阶段模型"""
    criterion = nn.MSELoss()
    
    # Stage1: 训练补全
    print("  Training Stage1 (Completion)...")
    optimizer_stage1 = torch.optim.Adam([
        {'params': model.completion_encoder.parameters()},
        {'params': model.completion_decoder.parameters()}
    ], lr=lr)
    
    best_loss_stage1 = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x_input, x_full, _ in train_loader:
            x_input = x_input.to(device)
            x_full = x_full.to(device)
            x_future = x_full[:, model.input_length:, :]
            
            optimizer_stage1.zero_grad()
            _, pred_future = model.complete_sequence(x_input)
            loss = criterion(pred_future, x_future)
            loss.backward()
            optimizer_stage1.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Completion Loss: {train_loss:.6f}")
        
        if train_loss < best_loss_stage1:
            best_loss_stage1 = train_loss
    
    model.stage1_trained = True
    
    # Stage2: 训练回归
    print("  Training Stage2 (Regression)...")
    
    # 冻结Stage1
    for param in model.completion_encoder.parameters():
        param.requires_grad = False
    for param in model.completion_decoder.parameters():
        param.requires_grad = False
    
    optimizer_stage2 = torch.optim.Adam([
        {'params': model.regression_encoder.parameters()},
        {'params': model.regressor.parameters()}
    ], lr=lr)
    
    best_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x_input, _, y_yield in train_loader:
            x_input = x_input.to(device)
            y_yield = y_yield.to(device)
            
            optimizer_stage2.zero_grad()
            y_pred = model(x_input)
            loss = criterion(y_pred, y_yield)
            loss.backward()
            optimizer_stage2.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_input, _, y_yield in test_loader:
                x_input = x_input.to(device)
                y_yield = y_yield.to(device)
                y_pred = model(x_input)
                loss = criterion(y_pred, y_yield)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")
    
    model.load_state_dict(best_state)
    model.stage2_trained = True
    
    return model, history


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_model(model, test_loader, yield_mean, yield_std, device='cuda'):
    """评估模型性能"""
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for x_input, _, y_yield in test_loader:
            x_input = x_input.to(device)
            y_pred = model(x_input)
            
            all_preds.extend(y_pred.cpu().numpy())
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
    mape = np.mean(np.abs((all_trues - all_preds) / (all_trues + 1e-6))) * 100
    
    # 残差
    residuals = all_trues - all_preds
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'predictions': all_preds,
        'targets': all_trues,
        'residuals': residuals
    }


def evaluate_completion_quality(model, test_loader, seq_mean, seq_std, device='cuda'):
    """评估补全质量（仅两阶段模型）"""
    if not isinstance(model, TwoStageRegressor):
        return None
    
    model.eval()
    
    all_pred_future = []
    all_true_future = []
    
    with torch.no_grad():
        for x_input, x_full, _ in test_loader:
            x_input = x_input.to(device)
            x_full = x_full.to(device)
            
            _, pred_future = model.complete_sequence(x_input)
            true_future = x_full[:, model.input_length:, :]
            
            all_pred_future.append(pred_future.cpu().numpy())
            all_true_future.append(true_future.cpu().numpy())
    
    all_pred_future = np.concatenate(all_pred_future, axis=0)
    all_true_future = np.concatenate(all_true_future, axis=0)
    
    # 反归一化
    all_pred_future = all_pred_future * seq_std[:, model.input_length:, :] + seq_mean[:, model.input_length:, :]
    all_true_future = all_true_future * seq_std[:, model.input_length:, :] + seq_mean[:, model.input_length:, :]
    
    # 计算补全质量
    seq_mse = mean_squared_error(
        all_true_future.reshape(-1), 
        all_pred_future.reshape(-1)
    )
    seq_rmse = np.sqrt(seq_mse)
    
    # 按时间步计算
    rmse_per_step = []
    for t in range(all_true_future.shape[1]):
        rmse_t = np.sqrt(mean_squared_error(
            all_true_future[:, t, :].flatten(),
            all_pred_future[:, t, :].flatten()
        ))
        rmse_per_step.append(rmse_t)
    
    # 相关系数
    correlation = np.corrcoef(
        all_true_future.reshape(-1),
        all_pred_future.reshape(-1)
    )[0, 1]
    
    return {
        'seq_rmse': float(seq_rmse),
        'seq_mae': float(np.mean(np.abs(all_true_future - all_pred_future))),
        'correlation': float(correlation),
        'rmse_per_step': [float(x) for x in rmse_per_step]
    }


# ============================================================================
# 实验A: 补全 vs 不补全
# ============================================================================

def experiment_A_completion_vs_direct(
    train_files, test_files, selected_bands,
    input_lengths=[6, 12, 18, 24],
    target_length=36,
    device='cuda'
):
    """
    实验A: 对比补全和不补全
    """
    print("\n" + "="*70)
    print("实验A: 补全 vs 不补全")
    print("="*70)
    
    results = {}
    
    for input_len in input_lengths:
        print(f"\n{'─'*70}")
        print(f"输入长度: {input_len}个月")
        print(f"{'─'*70}")
        
        # 加载数据
        train_dataset = FlexibleYieldDataset(
            train_files, selected_bands, input_len, target_length
        )
        test_dataset = FlexibleYieldDataset(
            test_files, selected_bands, input_len, target_length
        )
        test_dataset.seq_mean = train_dataset.seq_mean
        test_dataset.seq_std = train_dataset.seq_std
        test_dataset.yield_mean = train_dataset.yield_mean
        test_dataset.yield_std = train_dataset.yield_std
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 方法1: 直接回归
        print("\n[1/2] 训练直接回归模型...")
        model_direct = DirectRegressor(input_len).to(device)
        model_direct, history_direct = train_direct_model(
            model_direct, train_loader, test_loader, epochs=50, device=device
        )
        results_direct = evaluate_model(
            model_direct, test_loader, 
            train_dataset.yield_mean, train_dataset.yield_std,
            device
        )
        
        # 方法2: 两阶段（补全到36月）
        print("\n[2/2] 训练两阶段模型...")
        model_twostage = TwoStageRegressor(input_len, target_length).to(device)
        model_twostage, history_twostage = train_twostage_model(
            model_twostage, train_loader, test_loader, epochs=50, device=device
        )
        results_twostage = evaluate_model(
            model_twostage, test_loader,
            train_dataset.yield_mean, train_dataset.yield_std,
            device
        )
        
        # 评估补全质量
        completion_quality = evaluate_completion_quality(
            model_twostage, test_loader,
            train_dataset.seq_mean, train_dataset.seq_std,
            device
        )
        
        # 保存结果
        results[input_len] = {
            'direct': results_direct,
            'twostage': results_twostage,
            'completion_quality': completion_quality,
            'history_direct': history_direct,
            'history_twostage': history_twostage
        }
        
        # 打印对比
        print(f"\n结果对比:")
        print(f"{'─'*70}")
        print(f"{'方法':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE':>10}")
        print(f"{'─'*70}")
        print(f"{'直接回归':<20} {results_direct['rmse']:>10.4f} {results_direct['mae']:>10.4f} {results_direct['r2']:>10.4f} {results_direct['mape']:>9.2f}%")
        print(f"{'两阶段（补全）':<20} {results_twostage['rmse']:>10.4f} {results_twostage['mae']:>10.4f} {results_twostage['r2']:>10.4f} {results_twostage['mape']:>9.2f}%")
        
        # 计算提升
        improvement = (results_twostage['rmse'] - results_direct['rmse']) / results_twostage['rmse'] * 100
        print(f"\n直接法相比两阶段: {improvement:+.2f}% RMSE")
        
        if improvement > 5:
            print("✅ 直接法显著更好")
        elif improvement < -5:
            print("❌ 两阶段法显著更好")
        else:
            print("⚖ 两者性能接近")
        
        if completion_quality:
            print(f"\n补全质量:")
            print(f"  序列RMSE: {completion_quality['seq_rmse']:.4f}")
            print(f"  相关系数: {completion_quality['correlation']:.4f}")
    
    return results


# ============================================================================
# 实验B: 不同补全长度
# ============================================================================

def experiment_B_completion_lengths(
    train_files, test_files, selected_bands,
    input_length=12,
    target_lengths=[18, 24, 30, 36],
    device='cuda'
):
    """
    实验B: 测试不同补全长度的影响
    """
    print("\n" + "="*70)
    print(f"实验B: 不同补全长度（输入={input_length}月）")
    print("="*70)
    
    results = {}
    
    # 基线：直接回归
    print(f"\n[基线] 训练直接回归模型（不补全）...")
    train_dataset = FlexibleYieldDataset(
        train_files, selected_bands, input_length, 36
    )
    test_dataset = FlexibleYieldDataset(
        test_files, selected_bands, input_length, 36
    )
    test_dataset.seq_mean = train_dataset.seq_mean
    test_dataset.seq_std = train_dataset.seq_std
    test_dataset.yield_mean = train_dataset.yield_mean
    test_dataset.yield_std = train_dataset.yield_std
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model_direct = DirectRegressor(input_length).to(device)
    model_direct, _ = train_direct_model(
        model_direct, train_loader, test_loader, epochs=50, device=device
    )
    results_direct = evaluate_model(
        model_direct, test_loader,
        train_dataset.yield_mean, train_dataset.yield_std,
        device
    )
    
    results['direct'] = results_direct
    
    # 测试不同补全长度
    for target_len in target_lengths:
        print(f"\n[补全→{target_len}月] 训练两阶段模型...")
        
        # 重新加载数据（目标长度不同）
        train_dataset = FlexibleYieldDataset(
            train_files, selected_bands, input_length, target_len
        )
        test_dataset = FlexibleYieldDataset(
            test_files, selected_bands, input_length, target_len
        )
        test_dataset.seq_mean = train_dataset.seq_mean
        test_dataset.seq_std = train_dataset.seq_std
        test_dataset.yield_mean = train_dataset.yield_mean
        test_dataset.yield_std = train_dataset.yield_std
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model_twostage = TwoStageRegressor(input_length, target_len).to(device)
        model_twostage, _ = train_twostage_model(
            model_twostage, train_loader, test_loader, epochs=50, device=device
        )
        results_twostage = evaluate_model(
            model_twostage, test_loader,
            train_dataset.yield_mean, train_dataset.yield_std,
            device
        )
        
        completion_quality = evaluate_completion_quality(
            model_twostage, test_loader,
            train_dataset.seq_mean, train_dataset.seq_std,
            device
        )
        
        results[target_len] = {
            'regression': results_twostage,
            'completion': completion_quality
        }
    
    # 打印对比
    print(f"\n{'='*70}")
    print("结果汇总")
    print(f"{'='*70}")
    print(f"{'补全长度':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'补全质量':>12}")
    print(f"{'─'*70}")
    print(f"{'不补全':<15} {results_direct['rmse']:>10.4f} {results_direct['mae']:>10.4f} {results_direct['r2']:>10.4f} {'N/A':>12}")
    
    for target_len in target_lengths:
        r = results[target_len]['regression']
        c = results[target_len]['completion']
        print(f"{'补全→' + str(target_len) + '月':<15} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['r2']:>10.4f} {c['seq_rmse']:>12.4f}")
    
    # 找到最优补全长度
    best_target = min(target_lengths, key=lambda t: results[t]['regression']['rmse'])
    print(f"\n最优补全长度: {best_target}月 (RMSE={results[best_target]['regression']['rmse']:.4f})")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*70)
    print("完整实验: 序列补全对回归的影响")
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
    
    # 创建结果目录
    os.makedirs('experiment_results', exist_ok=True)
    os.makedirs('experiment_figures', exist_ok=True)
    
    # 实验A: 补全 vs 不补全
    results_A = experiment_A_completion_vs_direct(
        train_files, test_files, selected_bands,
        input_lengths=[6, 12, 18, 24],
        device=device
    )
    
    # 保存实验A结果
    with open('experiment_results/experiment_A_results.json', 'w') as f:
        # 只保存可序列化的部分
        results_A_serializable = {}
        for k, v in results_A.items():
            results_A_serializable[str(k)] = {
                'direct': {
                    'rmse': v['direct']['rmse'],
                    'mae': v['direct']['mae'],
                    'r2': v['direct']['r2'],
                    'mape': v['direct']['mape']
                },
                'twostage': {
                    'rmse': v['twostage']['rmse'],
                    'mae': v['twostage']['mae'],
                    'r2': v['twostage']['r2'],
                    'mape': v['twostage']['mape']
                },
                'completion_quality': v['completion_quality']
            }
        json.dump(results_A_serializable, f, indent=2)
    
    # 实验B: 不同补全长度
    results_B = experiment_B_completion_lengths(
        train_files, test_files, selected_bands,
        input_length=12,
        target_lengths=[18, 24, 30, 36],
        device=device
    )
    
    # 保存实验B结果
    with open('experiment_results/experiment_B_results.json', 'w') as f:
        results_B_serializable = {}
        for k, v in results_B.items():
            if k == 'direct':
                results_B_serializable[str(k)] = {
                    'rmse': v['rmse'],
                    'mae': v['mae'],
                    'r2': v['r2']
                }
            else:
                results_B_serializable[str(k)] = {
                    'regression': {
                        'rmse': v['regression']['rmse'],
                        'mae': v['regression']['mae'],
                        'r2': v['regression']['r2']
                    },
                    'completion': v['completion']
                }
        json.dump(results_B_serializable, f, indent=2)
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    print("\n结果已保存到:")
    print("  - experiment_results/experiment_A_results.json")
    print("  - experiment_results/experiment_B_results.json")
    
    print("\n运行可视化:")
    print("  python visualize_results.py")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/3600:.1f}小时")

