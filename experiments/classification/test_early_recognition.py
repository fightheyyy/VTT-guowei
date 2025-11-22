"""
测试早期识别时间
找到首次F1≥0.8的时间点
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from experiments.classification.data_loader_classification_cached import create_classification_dataloaders_cached
from models.timesclip_classifier import LanguageOnlyTimesCLIPClassifier


def evaluate_at_time_steps(model, data_loader, device, time_steps, total_steps=37):
    """
    评估模型在特定时间步数的性能
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            # 截断到指定时间步
            if time_steps < total_steps:
                x_truncated = x[:, :time_steps, :].clone()
                # 用零填充到原长度
                padding = torch.zeros(x.size(0), total_steps - time_steps, x.size(2)).to(device)
                x_truncated = torch.cat([x_truncated, padding], dim=1)
            else:
                x_truncated = x
            
            logits = model(x_truncated)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # 计算F1
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return f1_macro, f1_per_class


def find_early_recognition_time(model_path, csv_path, time_steps_list, f1_threshold=0.8):
    """
    寻找最早可识别时间
    """
    print("="*70)
    print("早期识别时间测试")
    print("="*70)
    print(f"模型: {model_path}")
    print(f"F1阈值: {f1_threshold}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print("\n加载数据...")
    _, _, test_loader, num_classes = create_classification_dataloaders_cached(
        csv_path=csv_path,
        batch_size=64,
        use_cache=True,
        load_to_memory=False
    )
    
    # 加载模型
    print("加载模型...")
    model = LanguageOnlyTimesCLIPClassifier(
        time_steps=37,
        n_variates=14,
        num_classes=num_classes
    ).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"最佳验证F1: {checkpoint.get('val_f1', 'N/A')}")
    
    # 测试不同时间长度
    print(f"\n{'='*70}")
    print(f"{'时间步':<8} {'天数':<8} {'F1 (macro)':<12} {'类别0':<8} {'类别1':<8} {'类别2':<8} {'类别3':<8} {'状态'}")
    print(f"{'-'*70}")
    
    results = []
    earliest_identifiable_step = None
    
    for steps in time_steps_list:
        days = steps * 10
        f1_macro, f1_per_class = evaluate_at_time_steps(model, test_loader, device, steps)
        
        identifiable = "✓ 可识别" if f1_macro >= f1_threshold else "✗ 未达标"
        
        print(f"{steps:<8} {days:<8} {f1_macro:<12.4f} ", end='')
        for f1 in f1_per_class:
            print(f"{f1:<8.4f} ", end='')
        print(f"{identifiable}")
        
        results.append({
            'steps': steps,
            'days': days,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class.tolist()
        })
        
        if earliest_identifiable_step is None and f1_macro >= f1_threshold:
            earliest_identifiable_step = steps
    
    print(f"{'='*70}")
    
    # 总结
    if earliest_identifiable_step:
        earliest_days = earliest_identifiable_step * 10
        print(f"\n🎯 最早可识别时间: {earliest_identifiable_step}步 ({earliest_days}天)")
        print(f"   F1分数: {results[time_steps_list.index(earliest_identifiable_step)]['f1_macro']:.4f}")
        
        # 分析每个类别
        print(f"\n每个类别的最早识别时间:")
        for class_idx in range(num_classes):
            for i, res in enumerate(results):
                if res['f1_per_class'][class_idx] >= f1_threshold:
                    print(f"  类别{class_idx}: {res['steps']}步 ({res['days']}天), F1={res['f1_per_class'][class_idx]:.4f}")
                    break
            else:
                print(f"  类别{class_idx}: 未达到阈值")
    else:
        print(f"\n❌ 未找到满足F1≥{f1_threshold}的时间点")
        max_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1_macro'])
        max_f1_result = results[max_f1_idx]
        print(f"   最高F1: {max_f1_result['f1_macro']:.4f} (在{max_f1_result['steps']}步/{max_f1_result['days']}天)")
    
    # 绘制曲线
    plot_early_recognition_curve(results, f1_threshold)
    
    return results, earliest_identifiable_step


def plot_early_recognition_curve(results, f1_threshold=0.8):
    """绘制早期识别曲线"""
    days_list = [r['days'] for r in results]
    f1_list = [r['f1_macro'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # F1曲线
    plt.plot(days_list, f1_list, 'o-', linewidth=2, markersize=8, label='F1 (macro)')
    plt.axhline(y=f1_threshold, color='r', linestyle='--', label=f'阈值 (F1={f1_threshold})')
    
    # 标记最早可识别点
    for i, (days, f1) in enumerate(zip(days_list, f1_list)):
        if f1 >= f1_threshold:
            plt.plot(days, f1, 'r*', markersize=20)
            plt.annotate(f'最早识别点\n{days}天\nF1={f1:.3f}', 
                        xy=(days, f1), 
                        xytext=(20, -20), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            break
    
    plt.xlabel('天数', fontsize=12)
    plt.ylabel('F1分数 (macro)', fontsize=12)
    plt.title('早期识别性能曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'experiments/classification/timesclip_improved/results/early_recognition_curve.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n曲线图已保存: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试早期识别时间')
    parser.add_argument('--model_path', type=str, 
                       default='experiments/classification/timesclip_improved/checkpoints/language_only_best.pth',
                       help='模型路径')
    parser.add_argument('--csv_path', type=str, default='../../data/2018four.csv', help='数据路径')
    parser.add_argument('--f1_threshold', type=float, default=0.8, help='F1阈值')
    
    args = parser.parse_args()
    
    # 测试的时间步列表
    time_steps_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 37]
    
    results, earliest_step = find_early_recognition_time(
        model_path=args.model_path,
        csv_path=args.csv_path,
        time_steps_list=time_steps_list,
        f1_threshold=args.f1_threshold
    )

