"""
实验结果可视化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)


def load_results():
    """加载实验结果"""
    with open('experiment_results/experiment_A_results.json', 'r') as f:
        results_A = json.load(f)
    
    with open('experiment_results/experiment_B_results.json', 'r') as f:
        results_B = json.load(f)
    
    return results_A, results_B


def plot_experiment_A(results_A):
    """可视化实验A: 补全 vs 不补全"""
    
    input_lengths = sorted([int(k) for k in results_A.keys()])
    
    rmse_direct = [results_A[str(l)]['direct']['rmse'] for l in input_lengths]
    rmse_twostage = [results_A[str(l)]['twostage']['rmse'] for l in input_lengths]
    
    mae_direct = [results_A[str(l)]['direct']['mae'] for l in input_lengths]
    mae_twostage = [results_A[str(l)]['twostage']['mae'] for l in input_lengths]
    
    r2_direct = [results_A[str(l)]['direct']['r2'] for l in input_lengths]
    r2_twostage = [results_A[str(l)]['twostage']['r2'] for l in input_lengths]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('实验A: 补全 vs 不补全', fontsize=16, fontweight='bold')
    
    # 1. RMSE对比
    ax = axes[0, 0]
    x = np.arange(len(input_lengths))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rmse_direct, width, label='直接回归', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, rmse_twostage, width, label='两阶段（补全）', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('输入长度（月）')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE对比（越低越好）')
    ax.set_xticks(x)
    ax.set_xticklabels(input_lengths)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. MAE对比
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, mae_direct, width, label='直接回归', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, mae_twostage, width, label='两阶段（补全）', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('输入长度（月）')
    ax.set_ylabel('MAE')
    ax.set_title('MAE对比（越低越好）')
    ax.set_xticks(x)
    ax.set_xticklabels(input_lengths)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. R²对比
    ax = axes[1, 0]
    bars1 = ax.bar(x - width/2, r2_direct, width, label='直接回归', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, r2_twostage, width, label='两阶段（补全）', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('输入长度（月）')
    ax.set_ylabel('R²')
    ax.set_title('R²对比（越高越好）')
    ax.set_xticks(x)
    ax.set_xticklabels(input_lengths)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 4. 相对提升
    ax = axes[1, 1]
    improvements = [(d - t) / t * 100 for d, t in zip(rmse_direct, rmse_twostage)]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    
    bars = ax.bar(x, improvements, width*2, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('输入长度（月）')
    ax.set_ylabel('相对提升（%）')
    ax.set_title('直接法相比两阶段的RMSE提升\n（正值=直接法更好）')
    ax.set_xticks(x)
    ax.set_xticklabels(input_lengths)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiment_figures/experiment_A_comparison.png', dpi=300, bbox_inches='tight')
    print("已保存: experiment_figures/experiment_A_comparison.png")
    plt.close()
    
    # 补全质量分析
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('补全质量分析', fontsize=16, fontweight='bold')
    
    # 补全RMSE vs 回归性能
    ax = axes[0]
    completion_rmse = [results_A[str(l)]['completion_quality']['seq_rmse'] for l in input_lengths]
    regression_rmse = rmse_twostage
    
    ax.scatter(completion_rmse, regression_rmse, s=200, alpha=0.6, c=input_lengths, cmap='viridis')
    for i, length in enumerate(input_lengths):
        ax.annotate(f'{length}月', (completion_rmse[i], regression_rmse[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    # 拟合趋势线
    z = np.polyfit(completion_rmse, regression_rmse, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(completion_rmse), max(completion_rmse), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'趋势线 (斜率={z[0]:.2f})')
    
    ax.set_xlabel('补全质量（序列RMSE）')
    ax.set_ylabel('回归性能（产量RMSE）')
    ax.set_title('补全质量 vs 回归性能')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 相关系数
    ax = axes[1]
    correlations = [results_A[str(l)]['completion_quality']['correlation'] for l in input_lengths]
    
    bars = ax.bar(x, correlations, width*2, color='#3498db', alpha=0.8)
    ax.set_xlabel('输入长度（月）')
    ax.set_ylabel('相关系数')
    ax.set_title('补全序列与真实序列的相关性')
    ax.set_xticks(x)
    ax.set_xticklabels(input_lengths)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('experiment_figures/completion_quality_analysis.png', dpi=300, bbox_inches='tight')
    print("已保存: experiment_figures/completion_quality_analysis.png")
    plt.close()


def plot_experiment_B(results_B):
    """可视化实验B: 不同补全长度"""
    
    target_lengths = sorted([int(k) for k in results_B.keys() if k != 'direct'])
    
    rmse_values = [results_B['direct']['rmse']] + [results_B[str(l)]['regression']['rmse'] for l in target_lengths]
    mae_values = [results_B['direct']['mae']] + [results_B[str(l)]['regression']['mae'] for l in target_lengths]
    r2_values = [results_B['direct']['r2']] + [results_B[str(l)]['regression']['r2'] for l in target_lengths]
    
    labels = ['不补全'] + [f'补全→{l}月' for l in target_lengths]
    x_pos = np.arange(len(labels))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('实验B: 不同补全长度的影响（输入=12月）', fontsize=16, fontweight='bold')
    
    # 1. RMSE
    ax = axes[0]
    colors = ['#2ecc71'] + ['#3498db'] * len(target_lengths)
    bars = ax.bar(x_pos, rmse_values, color=colors, alpha=0.8)
    ax.set_xlabel('补全策略')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs 补全长度')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 标记最优
    best_idx = np.argmin(rmse_values)
    bars[best_idx].set_color('#f39c12')
    bars[best_idx].set_alpha(1.0)
    ax.text(best_idx, rmse_values[best_idx] * 0.95, '★ 最优', 
            ha='center', fontsize=12, fontweight='bold', color='#f39c12')
    
    # 2. MAE
    ax = axes[1]
    bars = ax.bar(x_pos, mae_values, color=colors, alpha=0.8)
    ax.set_xlabel('补全策略')
    ax.set_ylabel('MAE')
    ax.set_title('MAE vs 补全长度')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. R²
    ax = axes[2]
    bars = ax.bar(x_pos, r2_values, color=colors, alpha=0.8)
    ax.set_xlabel('补全策略')
    ax.set_ylabel('R²')
    ax.set_title('R² vs 补全长度')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_figures/experiment_B_completion_lengths.png', dpi=300, bbox_inches='tight')
    print("已保存: experiment_figures/experiment_B_completion_lengths.png")
    plt.close()
    
    # 补全长度曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    
    completion_lengths = [12] + target_lengths  # 12是输入长度（不补全）
    rmse_curve = rmse_values
    
    ax.plot(completion_lengths, rmse_curve, 'o-', linewidth=2, markersize=10, 
            color='#3498db', label='RMSE')
    
    # 标记关键点
    best_idx = np.argmin(rmse_curve)
    ax.plot(completion_lengths[best_idx], rmse_curve[best_idx], 
            'r*', markersize=20, label='最优点')
    
    ax.set_xlabel('序列长度（月）', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_title('补全长度对回归性能的影响\n（输入=12月，测试不同补全目标）', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # 添加注释
    ax.annotate(f'最优: {completion_lengths[best_idx]}月\nRMSE: {rmse_curve[best_idx]:.4f}',
               xy=(completion_lengths[best_idx], rmse_curve[best_idx]),
               xytext=(20, 20), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiment_figures/completion_length_curve.png', dpi=300, bbox_inches='tight')
    print("已保存: experiment_figures/completion_length_curve.png")
    plt.close()


def generate_summary_table(results_A, results_B):
    """生成汇总表格"""
    
    # 实验A汇总
    print("\n" + "="*80)
    print("实验A汇总: 补全 vs 不补全")
    print("="*80)
    
    input_lengths = sorted([int(k) for k in results_A.keys()])
    
    print(f"\n{'输入':<8} {'方法':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'相对提升':>12}")
    print("-"*80)
    
    for length in input_lengths:
        direct = results_A[str(length)]['direct']
        twostage = results_A[str(length)]['twostage']
        
        improvement = (direct['rmse'] - twostage['rmse']) / twostage['rmse'] * 100
        
        print(f"{length}月    {'直接回归':<15} {direct['rmse']:>10.4f} {direct['mae']:>10.4f} {direct['r2']:>10.4f}")
        print(f"{'':8} {'两阶段':<15} {twostage['rmse']:>10.4f} {twostage['mae']:>10.4f} {twostage['r2']:>10.4f} {improvement:>11.1f}%")
        print("-"*80)
    
    # 计算平均提升
    avg_improvement = np.mean([
        (results_A[str(l)]['direct']['rmse'] - results_A[str(l)]['twostage']['rmse']) / 
        results_A[str(l)]['twostage']['rmse'] * 100
        for l in input_lengths
    ])
    
    print(f"\n平均相对提升: {avg_improvement:+.2f}%")
    
    if avg_improvement > 5:
        print("✅ 结论: 直接法显著优于两阶段法")
    elif avg_improvement < -5:
        print("❌ 结论: 两阶段法显著优于直接法")
    else:
        print("⚖ 结论: 两种方法性能接近")
    
    # 实验B汇总
    print("\n" + "="*80)
    print("实验B汇总: 不同补全长度（输入=12月）")
    print("="*80)
    
    target_lengths = sorted([int(k) for k in results_B.keys() if k != 'direct'])
    
    print(f"\n{'补全策略':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-"*80)
    
    # 不补全
    direct = results_B['direct']
    print(f"{'不补全（基线）':<20} {direct['rmse']:>10.4f} {direct['mae']:>10.4f} {direct['r2']:>10.4f}")
    
    # 不同补全长度
    best_rmse = direct['rmse']
    best_strategy = '不补全'
    
    for length in target_lengths:
        r = results_B[str(length)]['regression']
        print(f"{f'补全→{length}月':<20} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['r2']:>10.4f}")
        
        if r['rmse'] < best_rmse:
            best_rmse = r['rmse']
            best_strategy = f'补全→{length}月'
    
    print("-"*80)
    print(f"\n最优策略: {best_strategy} (RMSE={best_rmse:.4f})")


def main():
    print("加载实验结果...")
    results_A, results_B = load_results()
    
    print("\n生成可视化...")
    plot_experiment_A(results_A)
    plot_experiment_B(results_B)
    
    print("\n生成汇总表格...")
    generate_summary_table(results_A, results_B)
    
    print("\n" + "="*80)
    print("可视化完成！")
    print("="*80)
    print("\n生成的图表:")
    print("  - experiment_figures/experiment_A_comparison.png")
    print("  - experiment_figures/completion_quality_analysis.png")
    print("  - experiment_figures/experiment_B_completion_lengths.png")
    print("  - experiment_figures/completion_length_curve.png")


if __name__ == "__main__":
    import os
    os.makedirs('experiment_figures', exist_ok=True)
    main()

