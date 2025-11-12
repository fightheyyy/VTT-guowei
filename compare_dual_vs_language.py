"""
对比双模态模型 vs 纯语言模型的性能
读取已保存的结果JSON文件进行对比
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from datetime import datetime
import os
import argparse


def load_results(json_path):
    """加载结果JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_comparison(results_dual, results_lang, save_path):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 转换为numpy数组
    preds_dual = np.array(results_dual['predictions'])
    targets_dual = np.array(results_dual['targets'])
    preds_lang = np.array(results_lang['predictions'])
    targets_lang = np.array(results_lang['targets'])
    
    # 1. 散点图对比
    ax = axes[0, 0]
    ax.scatter(targets_dual, preds_dual, alpha=0.5, label='双模态', s=20, color='blue')
    ax.scatter(targets_lang, preds_lang, alpha=0.5, label='纯语言', s=20, color='orange')
    min_val = min(targets_dual.min(), targets_lang.min())
    max_val = max(targets_dual.max(), targets_lang.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    ax.set_xlabel('真实产量', fontsize=12)
    ax.set_ylabel('预测产量', fontsize=12)
    ax.set_title('预测 vs 真实对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 误差分布对比
    ax = axes[0, 1]
    errors_dual = preds_dual - targets_dual
    errors_lang = preds_lang - targets_lang
    ax.hist(errors_dual, bins=30, alpha=0.6, label='双模态', edgecolor='black', color='blue')
    ax.hist(errors_lang, bins=30, alpha=0.6, label='纯语言', edgecolor='black', color='orange')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='零误差')
    ax.set_xlabel('预测误差', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('误差分布对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. 指标对比柱状图
    ax = axes[1, 0]
    metrics = ['RMSE', 'MAE', 'R²', 'MAPE(%)']
    dual_vals = [
        results_dual['rmse'], 
        results_dual['mae'], 
        results_dual['r2'], 
        results_dual['mape']
    ]
    lang_vals = [
        results_lang['rmse'], 
        results_lang['mae'], 
        results_lang['r2'], 
        results_lang['mape']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, dual_vals, width, label='双模态', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, lang_vals, width, label='纯语言', alpha=0.8, color='orange')
    
    # 在柱子上标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('评估指标对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 性能提升百分比
    ax = axes[1, 1]
    improvements = []
    improvement_labels = []
    
    for metric, dual_val, lang_val in zip(metrics, dual_vals, lang_vals):
        if lang_val == 0 or (metric == 'MAPE(%)' and lang_val == float('inf')):
            continue
            
        if metric == 'R²':
            # R²越大越好，使用差值而不是百分比（因为可能为负）
            # 如果两个都是负数，直接看谁离0更近
            if dual_val > lang_val:
                improve = abs(dual_val - lang_val) / (1.0 + abs(lang_val)) * 100  # 归一化
            else:
                improve = -abs(dual_val - lang_val) / (1.0 + abs(lang_val)) * 100
        else:
            # 其他指标越小越好
            improve = ((lang_val - dual_val) / abs(lang_val)) * 100 if lang_val != 0 else 0
        
        improvements.append(improve)
        improvement_labels.append(metric)
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.barh(improvement_labels, improvements, color=colors, alpha=0.7)
    
    # 在柱子上标注数值
    for bar, val in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{val:+.1f}%', ha='left' if width > 0 else 'right', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('双模态相对纯语言的提升 (%)', fontsize=12)
    ax.set_title('性能提升百分比 (正值=双模态更好)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n对比可视化已保存: {save_path}")


def print_comparison_table(results_dual, results_lang):
    """打印对比表格"""
    print(f"\n{'='*70}")
    print(f"{'模型对比结果':^70}")
    print(f"{'='*70}")
    print(f"{'指标':<15} {'双模态':>15} {'纯语言':>15} {'双模态更优':>15}")
    print(f"{'-'*70}")
    
    metrics = [
        ('RMSE', results_dual['rmse'], results_lang['rmse'], False),
        ('MAE', results_dual['mae'], results_lang['mae'], False),
        ('R²', results_dual['r2'], results_lang['r2'], True),
        ('MAPE(%)', results_dual['mape'], results_lang['mape'], False)
    ]
    
    for name, dual_val, lang_val, higher_better in metrics:
        if lang_val == 0 or (name == 'MAPE(%)' and lang_val == float('inf')):
            improve_str = 'N/A'
        else:
            if higher_better:
                # R²：双模态更大则更好
                is_better = dual_val > lang_val
                if dual_val > lang_val:
                    improve = abs(dual_val - lang_val) / (1.0 + abs(lang_val)) * 100
                else:
                    improve = -abs(dual_val - lang_val) / (1.0 + abs(lang_val)) * 100
            else:
                # RMSE/MAE/MAPE：双模态更小则更好
                is_better = dual_val < lang_val
                improve = ((lang_val - dual_val) / abs(lang_val)) * 100
            
            symbol = '✓' if is_better else '✗'
            improve_str = f"{improve:+.2f}% {symbol}"
        
        print(f"{name:<15} {dual_val:>15.4f} {lang_val:>15.4f} {improve_str:>15}")
    
    # 添加总结
    better_count = sum([
        results_dual['rmse'] < results_lang['rmse'],
        results_dual['mae'] < results_lang['mae'],
        results_dual['r2'] > results_lang['r2'],
        results_dual['mape'] < results_lang['mape']
    ])
    
    print(f"{'-'*70}")
    if better_count >= 3:
        print(f"{'结论: 双模态模型更优 (' + str(better_count) + '/4 指标更好)':^70}")
    elif better_count <= 1:
        print(f"{'结论: 纯语言模型更优 (' + str(4-better_count) + '/4 指标更好)':^70}")
    else:
        print(f"{'结论: 两者接近 (各有' + str(better_count) + '个和' + str(4-better_count) + '个指标更好)':^70}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='对比双模态vs纯语言模型')
    parser.add_argument('--dual_results', type=str, 
                       default='experiments/yield_prediction/timesclip/results/TimesCLIP_Full_steps12.json',
                       help='双模态结果JSON路径')
    parser.add_argument('--lang_results', type=str,
                       default='experiments/yield_prediction/timesclip/results/LanguageOnly_TimesCLIP_steps12.json',
                       help='纯语言结果JSON路径')
    parser.add_argument('--output_dir', type=str,
                       default='experiments/yield_prediction/timesclip/comparison',
                       help='对比结果保存目录')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"TimesCLIP 模型对比分析")
    print(f"{'='*70}")
    print(f"双模态结果: {args.dual_results}")
    print(f"纯语言结果: {args.lang_results}")
    print(f"{'='*70}\n")
    
    # 检查文件是否存在
    if not os.path.exists(args.dual_results):
        print(f"错误：双模态结果文件不存在: {args.dual_results}")
        print("请先训练双模态模型！")
        return
    
    if not os.path.exists(args.lang_results):
        print(f"错误：纯语言结果文件不存在: {args.lang_results}")
        print("请先训练纯语言模型！")
        return
    
    # 加载结果
    print("加载结果...")
    results_dual = load_results(args.dual_results)
    results_lang = load_results(args.lang_results)
    
    # 打印对比表格
    print_comparison_table(results_dual, results_lang)
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制对比图
    plot_path = os.path.join(args.output_dir, f'comparison_{timestamp}.png')
    plot_comparison(results_dual, results_lang, plot_path)
    
    # 保存对比JSON
    comparison_data = {
        'timestamp': timestamp,
        'dual_modal': {
            'rmse': results_dual['rmse'],
            'mae': results_dual['mae'],
            'r2': results_dual['r2'],
            'mape': results_dual['mape']
        },
        'language_only': {
            'rmse': results_lang['rmse'],
            'mae': results_lang['mae'],
            'r2': results_lang['r2'],
            'mape': results_lang['mape']
        },
        'improvements': {}
    }
    
    # 计算提升
    for metric in ['rmse', 'mae', 'mape']:
        dual_val = results_dual[metric]
        lang_val = results_lang[metric]
        if lang_val != 0 and lang_val != float('inf'):
            improve = ((lang_val - dual_val) / abs(lang_val)) * 100
            comparison_data['improvements'][metric] = f"{improve:+.2f}%"
    
    # R²单独处理（可能为负）
    dual_r2 = results_dual['r2']
    lang_r2 = results_lang['r2']
    if dual_r2 > lang_r2:
        improve_r2 = abs(dual_r2 - lang_r2) / (1.0 + abs(lang_r2)) * 100
    else:
        improve_r2 = -abs(dual_r2 - lang_r2) / (1.0 + abs(lang_r2)) * 100
    comparison_data['improvements']['r2'] = f"{improve_r2:+.2f}%"
    
    # 添加哪个模型更好的判断
    comparison_data['winner'] = {
        'rmse': '纯语言' if results_lang['rmse'] < results_dual['rmse'] else '双模态',
        'mae': '纯语言' if results_lang['mae'] < results_dual['mae'] else '双模态',
        'r2': '纯语言' if results_lang['r2'] > results_dual['r2'] else '双模态',
        'mape': '纯语言' if results_lang['mape'] < results_dual['mape'] else '双模态'
    }
    
    better_count = sum(1 for v in comparison_data['winner'].values() if v == '双模态')
    comparison_data['overall_winner'] = '双模态' if better_count >= 3 else ('纯语言' if better_count <= 1 else '接近')
    
    json_path = os.path.join(args.output_dir, f'comparison_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"对比结果已保存: {json_path}")
    
    print(f"\n{'='*70}")
    print(f"对比分析完成！")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

