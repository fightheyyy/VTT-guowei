"""
训练产量预测模型
测试不同输入天数对预测准确度的影响
"""

import os
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
from sklearn.metrics import r2_score, mean_absolute_error
import json

from models.simple_yield_predictor import SimpleYieldPredictor, LanguageOnlyYieldPredictor
from data_loader_yield import create_yield_dataloaders


def train_model(
    model,
    train_loader,
    test_loader,
    yield_mean,
    yield_std,
    epochs=50,
    lr=1e-4,
    device='cuda',
    log_dir='logs/yield_prediction'
):
    """训练产量预测模型"""
    
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
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                
                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        val_loss /= len(test_loader)
        
        # 反归一化计算真实指标
        all_preds = np.array(all_preds) * yield_std + yield_mean
        all_targets = np.array(all_targets) * yield_std + yield_mean
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-6))) * 100
        
        # 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/RMSE', rmse, epoch)
        writer.add_scalar('Metrics/MAE', mae, epoch)
        writer.add_scalar('Metrics/R2', r2, epoch)
        writer.add_scalar('Metrics/MAPE', mape, epoch)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
            print(f"  ✓ 保存最佳模型")
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}: Val Loss={val_loss:.6f} (patience={patience_counter}/{max_patience})")
        
        if patience_counter >= max_patience:
            print(f"\n早停：{max_patience}轮无改善")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    writer.close()
    
    return model


def evaluate_model(model, test_loader, yield_mean, yield_std, device='cuda'):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_pred = model(x)
            
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 反归一化
    all_preds = all_preds * yield_std + yield_mean
    all_targets = all_targets * yield_std + yield_mean
    
    # 计算指标
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-6))) * 100
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'predictions': all_preds.flatten(),
        'targets': all_targets.flatten()
    }


def experiment_input_length_impact(
    train_files,
    test_files,
    selected_bands,
    input_steps_list=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
    model_type='language_only',  # 'both' or 'language_only'
    epochs=50,
    device='cuda'
):
    """
    实验：不同输入长度对预测准确度的影响
    """
    print("="*70)
    print(f"实验：输入天数对产量预测准确度的影响")
    print(f"模型类型: {model_type}")
    print("="*70)
    
    results = {}
    os.makedirs('checkpoints/yield_prediction', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    for input_steps in input_steps_list:
        days = input_steps * 10
        print(f"\n{'='*70}")
        print(f"测试：{input_steps}步 = {days}天")
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
        if model_type == 'both':
            model = SimpleYieldPredictor(
                time_steps=input_steps,
                n_variates=n_variates,
                d_model=256,
                use_vision=True,
                use_language=True,
                patch_length=min(6, input_steps),
                stride=min(3, input_steps // 2)
            ).to(device)
        else:  # language_only
            model = LanguageOnlyYieldPredictor(
                time_steps=input_steps,
                n_variates=n_variates,
                d_model=256,
                patch_length=min(6, input_steps),
                stride=min(3, input_steps // 2)
            ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数: {total_params:,}")
        
        # 训练
        model = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            yield_mean=yield_mean,
            yield_std=yield_std,
            epochs=epochs,
            lr=1e-4,
            device=device,
            log_dir=f'logs/yield_prediction/{model_type}_steps{input_steps}'
        )
        
        # 评估
        metrics = evaluate_model(model, test_loader, yield_mean, yield_std, device)
        
        print(f"\n最终结果 ({days}天):")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        # 保存结果
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
            f'checkpoints/yield_prediction/{model_type}_steps{input_steps}.pth'
        )
    
    # 保存所有结果
    with open(f'results/{model_type}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def visualize_results(results, model_type='language_only'):
    """可视化结果"""
    
    input_steps = sorted([int(k) for k in results.keys()])
    days = [results[str(s)]['days'] for s in input_steps]
    rmse_values = [results[str(s)]['rmse'] for s in input_steps]
    mae_values = [results[str(s)]['mae'] for s in input_steps]
    r2_values = [results[str(s)]['r2'] for s in input_steps]
    mape_values = [results[str(s)]['mape'] for s in input_steps]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'输入天数对产量预测准确度的影响\n({model_type})', 
                 fontsize=16, fontweight='bold')
    
    # 1. RMSE vs 天数
    ax = axes[0, 0]
    ax.plot(days, rmse_values, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('RMSE（越低越好）', fontsize=12)
    ax.set_title('RMSE vs 输入天数')
    ax.grid(True, alpha=0.3)
    
    # 标记最优点
    best_idx = np.argmin(rmse_values)
    ax.plot(days[best_idx], rmse_values[best_idx], 'r*', markersize=20)
    ax.annotate(f'最优: {days[best_idx]}天\nRMSE={rmse_values[best_idx]:.4f}',
               xy=(days[best_idx], rmse_values[best_idx]),
               xytext=(20, 20), textcoords='offset points',
               bbox=dict(boxstyle='round', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. MAE vs 天数
    ax = axes[0, 1]
    ax.plot(days, mae_values, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('MAE（越低越好）', fontsize=12)
    ax.set_title('MAE vs 输入天数')
    ax.grid(True, alpha=0.3)
    
    # 3. R² vs 天数
    ax = axes[1, 0]
    ax.plot(days, r2_values, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('R²（越高越好）', fontsize=12)
    ax.set_title('R² vs 输入天数')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 4. MAPE vs 天数
    ax = axes[1, 1]
    ax.plot(days, mape_values, 'o-', linewidth=2, markersize=8, color='#9b59b6')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('MAPE %（越低越好）', fontsize=12)
    ax.set_title('MAPE vs 输入天数')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_type}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: results/{model_type}_analysis.png")
    plt.close()
    
    # 汇总表格
    print(f"\n{'='*80}")
    print("结果汇总")
    print(f"{'='*80}")
    print(f"{'天数':<8} {'步数':<8} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE':>10}")
    print("-"*80)
    
    for i, s in enumerate(input_steps):
        print(f"{days[i]:>4}天  {s:>3}步    {rmse_values[i]:>10.4f} {mae_values[i]:>10.4f} "
              f"{r2_values[i]:>10.4f} {mape_values[i]:>9.2f}%")
    
    print("-"*80)
    print(f"\n最优配置: {days[best_idx]}天（{input_steps[best_idx]}步）")
    print(f"  RMSE: {rmse_values[best_idx]:.4f}")
    print(f"  MAE:  {mae_values[best_idx]:.4f}")
    print(f"  R²:   {r2_values[best_idx]:.4f}")
    print(f"  MAPE: {mape_values[best_idx]:.2f}%")
    
    # 性能改善分析
    print(f"\n{'='*80}")
    print("性能改善分析")
    print(f"{'='*80}")
    
    baseline_rmse = rmse_values[0]  # 最短天数作为基线
    
    print(f"基线（{days[0]}天）RMSE: {baseline_rmse:.4f}")
    print(f"\n相比基线的改善:")
    for i, s in enumerate(input_steps[1:], 1):
        improvement = (baseline_rmse - rmse_values[i]) / baseline_rmse * 100
        print(f"  {days[i]:>4}天: {improvement:+6.2f}% RMSE")
    
    # 边际效益分析
    print(f"\n边际效益（增加10天的改善）:")
    for i in range(1, len(days)):
        if days[i] - days[i-1] == 10:
            marginal = (rmse_values[i-1] - rmse_values[i]) / rmse_values[i-1] * 100
            print(f"  {days[i-1]:>4}天→{days[i]:>4}天: {marginal:+6.2f}% RMSE")


def main():
    # 配置
    train_files = [
        "extract2019_20251010_165007.csv",
        "extract2020_20251010_165007.csv",
        "extract2021_20251010_165007.csv"
    ]
    test_files = [
        "extract2022_20251010_165007.csv"
    ]
    
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 实验：不同输入长度
    results = experiment_input_length_impact(
        train_files=train_files,
        test_files=test_files,
        selected_bands=selected_bands,
        input_steps_list=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
        model_type='language_only',  # 先用语言模态（更快）
        epochs=50,
        device=device
    )
    
    # 可视化
    visualize_results(results, model_type='language_only')
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - results/language_only_results.json")
    print("  - results/language_only_analysis.png")
    print("  - checkpoints/yield_prediction/*.pth")
    print("\n查看训练曲线:")
    print("  tensorboard --logdir=logs/yield_prediction")


if __name__ == "__main__":
    main()

