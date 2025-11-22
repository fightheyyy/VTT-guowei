"""
对比不同时间步长度的分类效果
测试: 6步(60天) vs 12步(120天) vs 18步(180天) vs 37步(370天)
"""
import sys
sys.path.append('../..')

from experiments.classification.train_classification_improved import train_timesclip_classifier_improved
import matplotlib.pyplot as plt
import json

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    print("="*70)
    print("对比不同时间步长度的分类效果")
    print("="*70)
    
    # 测试的时间步列表
    test_timesteps = [6, 12, 18, 37]
    results = {}
    
    for ts in test_timesteps:
        print(f"\n{'='*70}")
        print(f"训练 {ts} 步 ({ts*10} 天) 模型")
        print(f"{'='*70}\n")
        
        try:
            model, metrics = train_timesclip_classifier_improved(
                csv_path="../../data/2018four.csv",
                time_steps=ts,
                n_variates=14,
                model_type="dual",
                batch_size=64,
                epochs=50,  # 减少epoch加快对比
                lr=1e-4,
                patience=10,
                
                # 改进策略
                focal_alpha=0.25,
                focal_gamma=2.0,
                time_weight_factor=3.0,
                focus_early=False,  # 关闭课程学习
                
                use_cached_images=False,
                test_only=False
            )
            
            results[ts] = {
                'f1_macro': metrics['f1_macro'],
                'accuracy': metrics['accuracy'],
                'f1_per_class': metrics['f1_per_class'],
                'days': ts * 10
            }
            
            print(f"\n{ts}步结果: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"训练{ts}步时出错: {e}")
            continue
    
    # 保存结果
    with open('experiments/classification/timesteps_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 可视化对比
    if len(results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        timesteps_list = sorted(results.keys())
        f1_scores = [results[ts]['f1_macro'] for ts in timesteps_list]
        accuracies = [results[ts]['accuracy'] for ts in timesteps_list]
        days_list = [ts * 10 for ts in timesteps_list]
        
        # F1对比
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(timesteps_list)), f1_scores, color='steelblue', alpha=0.8)
        ax1.set_xlabel('时间步 (天数)', fontsize=12)
        ax1.set_ylabel('F1 Score (macro)', fontsize=12)
        ax1.set_title('不同时间步的F1对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(timesteps_list)))
        ax1.set_xticklabels([f'{ts}步\n({days}天)' for ts, days in zip(timesteps_list, days_list)])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (bar, f1) in enumerate(zip(bars1, f1_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, f1 + 0.01, 
                    f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 准确率对比
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(timesteps_list)), accuracies, color='coral', alpha=0.8)
        ax2.set_xlabel('时间步 (天数)', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_title('不同时间步的准确率对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(timesteps_list)))
        ax2.set_xticklabels([f'{ts}步\n({days}天)' for ts, days in zip(timesteps_list, days_list)])
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, acc) in enumerate(zip(bars2, accuracies)):
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('experiments/classification/timesteps_comparison.png', 
                   dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存: experiments/classification/timesteps_comparison.png")
        
        # 打印总结
        print("\n" + "="*70)
        print("对比总结")
        print("="*70)
        best_ts = max(results.keys(), key=lambda ts: results[ts]['f1_macro'])
        print(f"最佳时间步: {best_ts}步 ({best_ts*10}天)")
        print(f"  F1 Score: {results[best_ts]['f1_macro']:.4f}")
        print(f"  准确率: {results[best_ts]['accuracy']:.4f}")
        print("\n所有结果:")
        for ts in timesteps_list:
            print(f"  {ts}步({ts*10}天): F1={results[ts]['f1_macro']:.4f}, "
                  f"Acc={results[ts]['accuracy']:.4f}")
        print("="*70)

