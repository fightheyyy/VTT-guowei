"""
对比实验: 语言模态 vs 双模态产量预测
目标: 对比纯语言模态和双模态在不同输入长度下的性能差距
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
from datetime import datetime

from models.simple_yield_predictor import LanguageOnlyYieldPredictor, SimpleYieldPredictor
from experiments.yield_prediction.data_loader import create_yield_dataloaders
import glob


def _resolve_data_paths():
    """
    解析数据文件路径：
    1) 优先使用 data/2019产量数据.csv 等标准命名
    2) 若不存在，则回退寻找 extractYYYY_*.csv（支持位于仓库根目录或 data/ 下）
    """
    # 首选：标准命名的数据文件
    train_default = [
        "data/2019产量数据.csv",
        "data/2020产量数据.csv",
        "data/2021产量数据.csv",
    ]
    test_default = ["data/2022产量数据.csv"]

    if all(os.path.exists(p) for p in train_default) and all(os.path.exists(p) for p in test_default):
        return train_default, test_default

    # 回退：查找 extractYYYY_*.csv（优先 data/ 目录，其次仓库根目录）
    def find_extract(year: int):
        patterns = [
            os.path.join("data", f"extract{year}_*.csv"),
            f"extract{year}_*.csv",
        ]
        for pat in patterns:
            matches = sorted(glob.glob(pat))
            if matches:
                return matches[-1]  # 选最新版本
        return None

    train_candidates = [find_extract(2019), find_extract(2020), find_extract(2021)]
    test_candidate = find_extract(2022)

    if all(m is not None for m in train_candidates) and test_candidate is not None:
        print("\n⚠ 未找到标准命名数据文件，已回退使用 extractYYYY_*.csv：")
        for p in train_candidates:
            print(f"  训练: {p}")
        print(f"  测试: {test_candidate}")
        return train_candidates, [test_candidate]

    # 两者都找不到，提示用户
    missing = train_default + test_default
    raise FileNotFoundError(
        "未找到产量数据文件。请按照以下任一方式准备数据：\n"
        "1) 将数据放在 data/ 目录，命名为：2019产量数据.csv, 2020产量数据.csv, 2021产量数据.csv, 2022产量数据.csv\n"
        "2) 或者将提取文件放在仓库根目录或 data/ 下，命名为：extract2019_*.csv, extract2020_*.csv, extract2021_*.csv, extract2022_*.csv\n"
        f"当前缺失: {', '.join(missing)}"
    )


def train_model(model, train_loader, test_loader, yield_mean, yield_std, 
                epochs=50, lr=1e-4, device='cuda', log_dir='logs', model_name='model'):
    """训练模型"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
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
        
        pbar = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/{epochs}', leave=False)
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
        r2 = r2_score(all_targets, all_preds)
        
        # 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/RMSE', rmse, epoch)
        writer.add_scalar('Metrics/R2', r2, epoch)
        
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, RMSE={rmse:.4f}, R²={r2:.4f} ✓")
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0:
                print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Val={val_loss:.4f} (patience={patience_counter}/{max_patience})")
        
        if patience_counter >= max_patience:
            print(f"  [{model_name}] 早停于Epoch {epoch+1}, 最佳Val={best_loss:.4f}")
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
    mask = all_targets > 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'predictions': all_preds.flatten().tolist(),
        'targets': all_targets.flatten().tolist()
    }


def run_comparison_experiment(input_steps_list=[6, 12, 18, 24, 30, 36], epochs=50):
    """运行对比实验"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("对比实验: 语言模态 vs 双模态产量预测")
    print("="*80)
    
    # 数据路径（带自动回退）
    train_files, test_files = _resolve_data_paths()
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n配置:")
    print(f"  设备: {device}")
    print(f"  训练数据: 2019-2021年（3年）")
    print(f"  测试数据: 2022年")
    print(f"  测试点: {input_steps_list}")
    print(f"  训练轮数: {epochs}")
    
    results_language = {}
    results_dual = {}
    
    os.makedirs('experiments/yield_prediction/comparison/checkpoints', exist_ok=True)
    os.makedirs('experiments/yield_prediction/comparison/results', exist_ok=True)
    
    for input_steps in input_steps_list:
        days = input_steps * 10
        print(f"\n{'='*80}")
        print(f"测试: {input_steps}步 = {days}天")
        print(f"{'='*80}")
        
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
        
        # ==================== 训练语言模态 ====================
        print(f"\n[1/2] 训练语言模态模型...")
        model_language = LanguageOnlyYieldPredictor(
            time_steps=input_steps,
            n_variates=n_variates,
            d_model=256,
            patch_length=min(6, input_steps),
            stride=min(3, max(1, input_steps // 2))
        ).to(device)
        
        model_language = train_model(
            model_language, train_loader, test_loader,
            yield_mean, yield_std,
            epochs=epochs, device=device,
            log_dir=f'experiments/yield_prediction/comparison/logs/language_steps{input_steps}',
            model_name='Language'
        )
        
        metrics_language = evaluate_model(model_language, test_loader, yield_mean, yield_std, device)
        
        print(f"\n  语言模态结果:")
        print(f"    RMSE: {metrics_language['rmse']:.4f}")
        print(f"    MAE:  {metrics_language['mae']:.4f}")
        print(f"    R²:   {metrics_language['r2']:.4f}")
        print(f"    MAPE: {metrics_language['mape']:.2f}%")
        
        # ==================== 训练双模态 ====================
        print(f"\n[2/2] 训练双模态模型...")
        model_dual = SimpleYieldPredictor(
            time_steps=input_steps,
            n_variates=n_variates,
            d_model=256,
            use_vision=True,
            clip_model_name="openai/clip-vit-base-patch16",
            patch_length=min(6, input_steps),
            stride=min(3, max(1, input_steps // 2))
        ).to(device)
        
        model_dual = train_model(
            model_dual, train_loader, test_loader,
            yield_mean, yield_std,
            epochs=epochs, device=device,
            log_dir=f'experiments/yield_prediction/comparison/logs/dual_steps{input_steps}',
            model_name='Dual'
        )
        
        metrics_dual = evaluate_model(model_dual, test_loader, yield_mean, yield_std, device)
        
        print(f"\n  双模态结果:")
        print(f"    RMSE: {metrics_dual['rmse']:.4f}")
        print(f"    MAE:  {metrics_dual['mae']:.4f}")
        print(f"    R²:   {metrics_dual['r2']:.4f}")
        print(f"    MAPE: {metrics_dual['mape']:.2f}%")
        
        # ==================== 对比 ====================
        rmse_diff = metrics_language['rmse'] - metrics_dual['rmse']
        r2_diff = metrics_language['r2'] - metrics_dual['r2']
        
        print(f"\n  对比 (语言 vs 双模态):")
        print(f"    RMSE差异: {rmse_diff:+.4f} ({'语言更好' if rmse_diff < 0 else '双模态更好'})")
        print(f"    R²差异:   {r2_diff:+.4f} ({'语言更好' if r2_diff > 0 else '双模态更好'})")
        
        # 保存结果
        results_language[input_steps] = {
            'days': days,
            'rmse': metrics_language['rmse'],
            'mae': metrics_language['mae'],
            'r2': metrics_language['r2'],
            'mape': metrics_language['mape']
        }
        
        results_dual[input_steps] = {
            'days': days,
            'rmse': metrics_dual['rmse'],
            'mae': metrics_dual['mae'],
            'r2': metrics_dual['r2'],
            'mape': metrics_dual['mape']
        }
        
        # 保存模型
        torch.save(
            model_language.state_dict(),
            f'experiments/yield_prediction/comparison/checkpoints/language_steps{input_steps}.pth'
        )
        torch.save(
            model_dual.state_dict(),
            f'experiments/yield_prediction/comparison/checkpoints/dual_steps{input_steps}.pth'
        )
    
    # 保存完整结果
    comparison_results = {
        'timestamp': timestamp,
        'config': {
            'input_steps_list': input_steps_list,
            'epochs': epochs,
            'device': device
        },
        'language_only': results_language,
        'dual_modal': results_dual
    }
    
    with open('experiments/yield_prediction/comparison/results/comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # 可视化对比
    visualize_comparison(results_language, results_dual)
    
    return comparison_results


def visualize_comparison(results_language, results_dual):
    """可视化对比结果"""
    
    input_steps = sorted(results_language.keys())
    days = [results_language[s]['days'] for s in input_steps]
    
    # 提取指标
    rmse_lang = [results_language[s]['rmse'] for s in input_steps]
    rmse_dual = [results_dual[s]['rmse'] for s in input_steps]
    r2_lang = [results_language[s]['r2'] for s in input_steps]
    r2_dual = [results_dual[s]['r2'] for s in input_steps]
    mae_lang = [results_language[s]['mae'] for s in input_steps]
    mae_dual = [results_dual[s]['mae'] for s in input_steps]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE对比
    ax = axes[0, 0]
    ax.plot(days, rmse_lang, 'o-', linewidth=2, markersize=8, color='#3498db', label='语言模态')
    ax.plot(days, rmse_dual, 's-', linewidth=2, markersize=8, color='#e74c3c', label='双模态')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE对比 (越低越好)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # R²对比
    ax = axes[0, 1]
    ax.plot(days, r2_lang, 'o-', linewidth=2, markersize=8, color='#3498db', label='语言模态')
    ax.plot(days, r2_dual, 's-', linewidth=2, markersize=8, color='#e74c3c', label='双模态')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('R²对比 (越高越好)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # MAE对比
    ax = axes[1, 0]
    ax.plot(days, mae_lang, 'o-', linewidth=2, markersize=8, color='#3498db', label='语言模态')
    ax.plot(days, mae_dual, 's-', linewidth=2, markersize=8, color='#e74c3c', label='双模态')
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE对比 (越低越好)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 性能差异（RMSE）
    ax = axes[1, 1]
    rmse_diff = [rmse_lang[i] - rmse_dual[i] for i in range(len(days))]
    colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in rmse_diff]
    bars = ax.bar(days, rmse_diff, color=colors, alpha=0.7, width=20)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('输入天数', fontsize=12)
    ax.set_ylabel('RMSE差异 (语言 - 双模态)', fontsize=12)
    ax.set_title('RMSE差异 (负值=语言更好, 正值=双模态更好)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加标注
    for bar, diff in zip(bars, rmse_diff):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig('experiments/yield_prediction/comparison/results/comparison.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存: experiments/yield_prediction/comparison/results/comparison.png")
    plt.close()
    
    # 生成对比表格
    print("\n" + "="*80)
    print("详细对比表格")
    print("="*80)
    print(f"{'天数':<8} {'语言RMSE':<12} {'双模态RMSE':<12} {'差异':<10} {'语言R²':<10} {'双模态R²':<10}")
    print("-"*80)
    for i, s in enumerate(input_steps):
        diff = rmse_lang[i] - rmse_dual[i]
        winner = "语言更好" if diff < 0 else "双模态更好"
        print(f"{days[i]:<8} {rmse_lang[i]:<12.4f} {rmse_dual[i]:<12.4f} {diff:+.4f} ({winner:<10}) "
              f"{r2_lang[i]:<10.4f} {r2_dual[i]:<10.4f}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='对比语言模态和双模态产量预测')
    parser.add_argument('--quick', action='store_true', help='快速测试（4个点）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    args = parser.parse_args()
    
    if args.quick:
        print("快速测试模式")
        input_steps_list = [6, 12, 18, 30]
        epochs = 30
    else:
        input_steps_list = [6, 12, 18, 24, 30, 36]
        epochs = args.epochs
    
    results = run_comparison_experiment(input_steps_list, epochs)
    
    print("\n" + "="*80)
    print("对比实验完成！")
    print("="*80)
    print("\n结果文件:")
    print("  - experiments/yield_prediction/comparison/results/comparison.json")
    print("  - experiments/yield_prediction/comparison/results/comparison.png")
    print("\n查看训练曲线:")
    print("  tensorboard --logdir=experiments/yield_prediction/comparison/logs")
    print("\n结论将保存在comparison.json中")

