"""
实验1: 产量预测 - 找到最短有效预测天数
目标: 测试不同输入长度（30-360天）对产量预测准确度的影响
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

from models.simple_yield_predictor import LanguageOnlyYieldPredictor
from experiments.yield_prediction.data_loader import create_yield_dataloaders


def train_model(model, train_loader, test_loader, yield_mean, yield_std, 
                epochs=50, lr=1e-4, device='cuda', log_dir='logs'):
    """训练模型"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    writer = SummaryWriter(log_dir)
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                
                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        val_loss /= len(test_loader)
        
        # 反归一化计算指标
        all_preds = np.array(all_preds) * yield_std + yield_mean
        all_targets = np.array(all_targets) * yield_std + yield_mean
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        # 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/RMSE', rmse, epoch)
        writer.add_scalar('Metrics/MAE', mae, epoch)
        writer.add_scalar('Metrics/R2', r2, epoch)
        
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            print(f"\nEpoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"RMSE={rmse:.4f}, R²={r2:.4f} ✓")
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Val={val_loss:.6f} (patience={patience_counter}/{max_patience})")
        
        if patience_counter >= max_patience:
            print(f"早停：{max_patience}轮无改善")
            break
    
    model.load_state_dict(best_model_state)
    writer.close()
    return model


def evaluate_model(model, test_loader, yield_mean, yield_std, device='cuda'):
    """评估模型"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_pred = model(x)
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.numpy())
    
    all_preds = np.array(all_preds) * yield_std + yield_mean
    all_targets = np.array(all_targets) * yield_std + yield_mean
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # MAPE: 只对非零值计算
    mask = all_targets > 0.1  # 过滤掉接近0的值
    if mask.sum() > 0:
        mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'predictions': all_preds.flatten(),
        'targets': all_targets.flatten()
    }


def run_experiment(input_steps_list=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
                   epochs=50):
    """运行完整实验"""
    
    print("="*70)
    print("实验1: 产量预测 - 最短有效预测天数")
    print("="*70)
    
    # 数据路径
    train_files = [
        "data/extract2019_20251010_165007.csv",
        "data/extract2020_20251010_165007.csv",
        "data/extract2021_20251010_165007.csv"
    ]
    test_files = ["data/extract2022_20251010_165007.csv"]
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print(f"训练数据: 2019-2021年（3年）")
    print(f"测试数据: 2022年")
    
    results = {}
    os.makedirs('experiments/yield_prediction/checkpoints', exist_ok=True)
    os.makedirs('experiments/yield_prediction/results', exist_ok=True)
    
    for input_steps in input_steps_list:
        days = input_steps * 10
        print(f"\n{'='*70}")
        print(f"测试: {input_steps}步 = {days}天")
        print(f"{'='*70}")
        
        # 加载数据
        train_loader, test_loader, n_variates = create_yield_dataloaders(
            train_csv_paths=train_files,
            test_csv_paths=test_files,
            selected_bands=selected_bands,
            input_steps=input_steps,
            batch_size=32
        )
        
        yield_mean = train_loader.dataset.yield_mean
        yield_std = train_loader.dataset.yield_std
        
        # 创建模型
        model = LanguageOnlyYieldPredictor(
            time_steps=input_steps,
            n_variates=n_variates,
            d_model=256,
            patch_length=min(6, input_steps),
            stride=min(3, input_steps // 2)
        ).to(device)
        
        # 训练
        model = train_model(
            model, train_loader, test_loader,
            yield_mean, yield_std,
            epochs=epochs, device=device,
            log_dir=f'experiments/yield_prediction/logs/steps{input_steps}'
        )
        
        # 评估
        metrics = evaluate_model(model, test_loader, yield_mean, yield_std, device)
        
        print(f"\n最终结果 ({days}天):")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        results[input_steps] = {
            'days': days,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'mape': metrics['mape']
        }
        
        # 保存模型
        torch.save(
            model.state_dict(),
            f'experiments/yield_prediction/checkpoints/model_steps{input_steps}.pth'
        )
    
    # 保存结果
    with open('experiments/yield_prediction/results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    visualize_results(results)
    
    return results


def visualize_results(results):
    """可视化结果"""
    
    input_steps = sorted([int(k) for k in results.keys()])
    days = [results[s]['days'] for s in input_steps]
    rmse_values = [results[s]['rmse'] for s in input_steps]
    r2_values = [results[s]['r2'] for s in input_steps]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE
    ax = axes[0]
    ax.plot(days, rmse_values, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    best_idx = np.argmin(rmse_values)
    ax.plot(days[best_idx], rmse_values[best_idx], 'r*', markersize=20)
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE vs 输入天数')
    ax.grid(True, alpha=0.3)
    
    # R²
    ax = axes[1]
    ax.plot(days, r2_values, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('R² vs 输入天数')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/yield_prediction/results/analysis.png', dpi=300)
    print("\n图表已保存: experiments/yield_prediction/results/analysis.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速测试（4个点）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    args = parser.parse_args()
    
    if args.quick:
        print("快速测试模式")
        input_steps_list = [6, 12, 18, 30]
        epochs = 30
    else:
        input_steps_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
        epochs = args.epochs
    
    results = run_experiment(input_steps_list, epochs)
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    print("\n结果文件:")
    print("  - experiments/yield_prediction/results/results.json")
    print("  - experiments/yield_prediction/results/analysis.png")
    print("\n查看训练曲线:")
    print("  tensorboard --logdir=experiments/yield_prediction/logs")

