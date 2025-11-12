"""
对比双模态和纯语言模型的分类性能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(model_type):
    """加载模型结果"""
    results_path = f"experiments/classification/timesclip/results/{model_type}_results.json"
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_models():
    """对比两个模型的性能"""
    
    # 加载结果
    dual_results = load_results('dual')
    lang_results = load_results('language_only')
    
    dual_metrics = dual_results['test_metrics']
    lang_metrics = lang_results['test_metrics']
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TimesCLIP分类任务: 双模态 vs 纯语言模态对比', fontsize=16, fontweight='bold')
    
    # 1. 指标对比柱状图
    ax = axes[0, 0]
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    dual_values = [dual_metrics[k] for k in metrics_keys]
    lang_values = [lang_metrics[k] for k in metrics_keys]
    
    bars1 = ax.bar(x - width/2, dual_values, width, label='双模态', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, lang_values, width, label='纯语言', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('分数')
    ax.set_title('性能指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 相对提升百分比
    ax = axes[0, 1]
    improvements = [(dual_values[i] - lang_values[i]) / lang_values[i] * 100 
                    for i in range(len(metrics_keys))]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(metrics_names, improvements, color=colors, alpha=0.7)
    ax.set_xlabel('相对提升 (%)')
    ax.set_title('双模态相比纯语言的提升')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax.text(imp, i, f'{imp:+.1f}%', ha='left' if imp > 0 else 'right', 
               va='center', fontsize=10, fontweight='bold')
    
    # 3. 混淆矩阵对比 - 双模态
    ax = axes[0, 2]
    dual_conf = np.array(dual_metrics['confusion_matrix'])
    sns.heatmap(dual_conf, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title('双模态混淆矩阵')
    ax.set_ylabel('真实类别')
    ax.set_xlabel('预测类别')
    
    # 4. 混淆矩阵对比 - 纯语言
    ax = axes[1, 0]
    lang_conf = np.array(lang_metrics['confusion_matrix'])
    sns.heatmap(lang_conf, annot=True, fmt='d', cmap='Reds', ax=ax, cbar=False)
    ax.set_title('纯语言混淆矩阵')
    ax.set_ylabel('真实类别')
    ax.set_xlabel('预测类别')
    
    # 5. 训练历史对比
    ax = axes[1, 1]
    dual_history = dual_results['train_history']
    lang_history = lang_results['train_history']
    
    dual_epochs = [h['epoch'] for h in dual_history]
    dual_val_acc = [h['val_acc'] for h in dual_history]
    lang_epochs = [h['epoch'] for h in lang_history]
    lang_val_acc = [h['val_acc'] for h in lang_history]
    
    ax.plot(dual_epochs, dual_val_acc, label='双模态', color='#3498db', linewidth=2)
    ax.plot(lang_epochs, lang_val_acc, label='纯语言', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('验证集准确率')
    ax.set_title('训练过程对比')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. 总结表格
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_data = [
        ['指标', '双模态', '纯语言', '提升'],
        ['准确率', f"{dual_metrics['accuracy']:.4f}", 
         f"{lang_metrics['accuracy']:.4f}", 
         f"{improvements[0]:+.1f}%"],
        ['精确率', f"{dual_metrics['precision']:.4f}", 
         f"{lang_metrics['precision']:.4f}", 
         f"{improvements[1]:+.1f}%"],
        ['召回率', f"{dual_metrics['recall']:.4f}", 
         f"{lang_metrics['recall']:.4f}", 
         f"{improvements[2]:+.1f}%"],
        ['F1分数', f"{dual_metrics['f1']:.4f}", 
         f"{lang_metrics['f1']:.4f}", 
         f"{improvements[3]:+.1f}%"],
        ['', '', '', ''],
        ['最佳验证准确率', f"{dual_results['best_val_acc']:.4f}", 
         f"{lang_results['best_val_acc']:.4f}", '']
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, 7):
        for j in range(4):
            if i == 6:  # 空行
                table[(i, j)].set_facecolor('#ecf0f1')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
    
    ax.set_title('性能对比总结', pad=20, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"experiments/classification/timesclip/comparison/comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: {save_path}")
    
    plt.show()
    
    # 打印详细对比
    print("\n" + "="*70)
    print("模型性能对比总结")
    print("="*70)
    print(f"\n{'指标':<15} {'双模态':<15} {'纯语言':<15} {'提升':<15}")
    print("-"*70)
    for i, name in enumerate(metrics_names):
        print(f"{name:<15} {dual_values[i]:<15.4f} {lang_values[i]:<15.4f} {improvements[i]:>+14.2f}%")
    
    print("\n最佳验证准确率:")
    print(f"  双模态: {dual_results['best_val_acc']:.4f}")
    print(f"  纯语言: {lang_results['best_val_acc']:.4f}")
    
    # 分析结论
    print("\n" + "="*70)
    print("分析结论:")
    print("="*70)
    
    avg_improvement = np.mean(improvements)
    if avg_improvement > 0:
        print(f"✓ 双模态模型平均提升: {avg_improvement:.2f}%")
        print("  双模态融合视觉和语言特征，显著提升了分类性能")
    elif avg_improvement < 0:
        print(f"✗ 双模态模型平均下降: {avg_improvement:.2f}%")
        print("  可能原因：")
        print("    - 视觉模态引入噪声")
        print("    - 特征融合策略需要优化")
        print("    - 对比学习权重需要调整")
    else:
        print("≈ 两个模型性能相当")
    
    print("="*70)


if __name__ == "__main__":
    import os
    
    # 创建对比结果文件夹
    os.makedirs("experiments/classification/timesclip/comparison", exist_ok=True)
    
    # 对比模型
    compare_models()

