"""
实验追踪系统
自动记录每次实验的配置、结果、对比分析
便于后续论文撰写
"""

import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ExperimentTracker:
    """
    实验追踪器
    用于记录和对比不同配置的训练结果
    """
    
    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.experiments_file = os.path.join(log_dir, "experiments.csv")
        
        # 如果不存在，创建实验记录表
        if not os.path.exists(self.experiments_file):
            self._init_experiments_table()
    
    def _init_experiments_table(self):
        """初始化实验记录表"""
        columns = [
            # 基本信息
            'experiment_id', 'timestamp', 'description', 'tags',
            
            # 数据配置
            'time_steps', 'train_size', 'val_size', 'test_size',
            
            # 模型配置
            'model_type', 'dropout', 'use_contrastive', 'contrastive_weight',
            
            # 训练配置
            'batch_size', 'lr', 'weight_decay', 'epochs',
            
            # 数据增强
            'augmentation_mode', 'ts_aug_prob', 'img_aug_prob', 'aug_types',
            
            # 损失函数
            'focal_gamma', 'focal_alpha',
            
            # 训练结果
            'best_epoch', 'best_train_f1', 'best_train_acc', 
            'best_val_f1', 'best_val_acc',
            'final_test_f1', 'final_test_acc',
            
            # 类别性能
            'class0_f1', 'class1_f1', 'class2_f1', 'class3_f1',
            
            # 过拟合指标
            'overfit_gap', 'train_val_f1_ratio',
            
            # 其他
            'total_params', 'trainable_params', 'training_time_hours',
            'notes'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.experiments_file, index=False)
    
    def log_experiment(self, config, results, notes=""):
        """
        记录一次实验
        
        Args:
            config: 配置字典
            results: 结果字典
            notes: 备注
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{timestamp}"
        
        # 准备记录数据
        record = {
            # 基本信息
            'experiment_id': exp_id,
            'timestamp': timestamp,
            'description': config.get('description', ''),
            'tags': ','.join(config.get('tags', [])),
            
            # 数据配置
            'time_steps': config.get('time_steps', 12),
            'train_size': results.get('train_size', 0),
            'val_size': results.get('val_size', 0),
            'test_size': results.get('test_size', 0),
            
            # 模型配置
            'model_type': config.get('model_type', 'dual'),
            'dropout': config.get('dropout', 0.1),
            'use_contrastive': config.get('use_contrastive', True),
            'contrastive_weight': config.get('contrastive_weight', 0.1),
            
            # 训练配置
            'batch_size': config.get('batch_size', 64),
            'lr': config.get('lr', 1e-4),
            'weight_decay': config.get('weight_decay', 1e-4),
            'epochs': config.get('epochs', 100),
            
            # 数据增强
            'augmentation_mode': config.get('augmentation_mode', 'none'),
            'ts_aug_prob': config.get('ts_aug_prob', 0),
            'img_aug_prob': config.get('img_aug_prob', 0),
            'aug_types': config.get('aug_types', ''),
            
            # 损失函数
            'focal_gamma': config.get('focal_gamma', 2.0),
            'focal_alpha': config.get('focal_alpha', 0.25),
            
            # 训练结果
            'best_epoch': results.get('best_epoch', 0),
            'best_train_f1': results.get('best_train_f1', 0),
            'best_train_acc': results.get('best_train_acc', 0),
            'best_val_f1': results.get('best_val_f1', 0),
            'best_val_acc': results.get('best_val_acc', 0),
            'final_test_f1': results.get('final_test_f1', 0),
            'final_test_acc': results.get('final_test_acc', 0),
            
            # 类别性能
            'class0_f1': results.get('class_f1', [0,0,0,0])[0],
            'class1_f1': results.get('class_f1', [0,0,0,0])[1],
            'class2_f1': results.get('class_f1', [0,0,0,0])[2],
            'class3_f1': results.get('class_f1', [0,0,0,0])[3],
            
            # 过拟合指标
            'overfit_gap': results.get('best_train_f1', 0) - results.get('best_val_f1', 0),
            'train_val_f1_ratio': results.get('best_train_f1', 0) / max(results.get('best_val_f1', 0.001), 0.001),
            
            # 其他
            'total_params': results.get('total_params', 0),
            'trainable_params': results.get('trainable_params', 0),
            'training_time_hours': results.get('training_time_hours', 0),
            'notes': notes
        }
        
        # 追加到CSV
        df = pd.read_csv(self.experiments_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(self.experiments_file, index=False)
        
        # 保存详细结果JSON
        detail_file = os.path.join(self.log_dir, f"{exp_id}_detail.json")
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': config,
                'results': results,
                'notes': notes
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 实验已记录: {exp_id}")
        print(f"  Val F1: {results.get('best_val_f1', 0):.4f}")
        print(f"  Test F1: {results.get('final_test_f1', 0):.4f}")
        print(f"  详细记录: {detail_file}")
        
        return exp_id
    
    def compare_experiments(self, exp_ids=None, tags=None, output_file="comparison_report.md"):
        """
        对比多个实验
        
        Args:
            exp_ids: 实验ID列表（如果为None，对比所有实验）
            tags: 标签过滤（如'augmentation', 'baseline'）
            output_file: 输出报告文件
        """
        df = pd.read_csv(self.experiments_file)
        
        # 过滤实验
        if exp_ids is not None:
            df = df[df['experiment_id'].isin(exp_ids)]
        
        if tags is not None:
            df = df[df['tags'].str.contains('|'.join(tags), na=False)]
        
        if len(df) == 0:
            print("没有找到匹配的实验")
            return
        
        # 生成对比报告
        report = self._generate_comparison_report(df)
        
        # 保存报告
        report_path = os.path.join(self.log_dir, output_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 对比报告已生成: {report_path}")
        
        # 生成对比图表
        self._generate_comparison_plots(df)
        
        return report_path
    
    def _generate_comparison_report(self, df):
        """生成Markdown格式的对比报告"""
        report = "# 实验对比报告\n\n"
        report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**对比实验数**: {len(df)}\n\n"
        report += "---\n\n"
        
        # 1. 性能排名
        report += "## 1. 性能排名\n\n"
        report += "### 按验证集F1排序\n\n"
        
        df_sorted = df.sort_values('best_val_f1', ascending=False)
        
        report += "| 排名 | 实验ID | 描述 | Val F1 | Test F1 | 过拟合差距 |\n"
        report += "|------|--------|------|--------|---------|------------|\n"
        
        for idx, row in enumerate(df_sorted.itertuples(), 1):
            medal = "🥇" if idx == 1 else ("🥈" if idx == 2 else ("🥉" if idx == 3 else ""))
            report += f"| {idx} {medal} | {row.experiment_id} | {row.description} | "
            report += f"{row.best_val_f1:.4f} | {row.final_test_f1:.4f} | "
            report += f"{row.overfit_gap:.4f} |\n"
        
        report += "\n"
        
        # 2. 增强策略对比
        if 'augmentation_mode' in df.columns:
            report += "## 2. 数据增强效果对比\n\n"
            
            aug_groups = df.groupby('augmentation_mode').agg({
                'best_val_f1': ['mean', 'std', 'max'],
                'overfit_gap': ['mean', 'std'],
                'experiment_id': 'count'
            }).round(4)
            
            report += aug_groups.to_markdown() + "\n\n"
        
        # 3. 超参数影响
        report += "## 3. 超参数影响分析\n\n"
        
        # Dropout影响
        report += "### Dropout vs Val F1\n\n"
        dropout_analysis = df.groupby('dropout')['best_val_f1'].agg(['mean', 'max', 'count'])
        report += dropout_analysis.to_markdown() + "\n\n"
        
        # Weight Decay影响
        report += "### Weight Decay vs Val F1\n\n"
        wd_analysis = df.groupby('weight_decay')['best_val_f1'].agg(['mean', 'max', 'count'])
        report += wd_analysis.to_markdown() + "\n\n"
        
        # 4. 最佳配置推荐
        report += "## 4. 最佳配置推荐\n\n"
        best_exp = df_sorted.iloc[0]
        
        report += f"**最佳实验**: {best_exp['experiment_id']}\n\n"
        report += f"- **验证集F1**: {best_exp['best_val_f1']:.4f}\n"
        report += f"- **测试集F1**: {best_exp['final_test_f1']:.4f}\n"
        report += f"- **过拟合差距**: {best_exp['overfit_gap']:.4f}\n\n"
        report += "**配置**:\n\n"
        report += f"- 数据增强: {best_exp['augmentation_mode']}\n"
        report += f"- Dropout: {best_exp['dropout']}\n"
        report += f"- Weight Decay: {best_exp['weight_decay']}\n"
        report += f"- 学习率: {best_exp['lr']}\n"
        report += f"- Batch Size: {best_exp['batch_size']}\n\n"
        
        # 5. 改进建议
        report += "## 5. 改进建议\n\n"
        
        avg_overfit_gap = df['overfit_gap'].mean()
        if avg_overfit_gap > 0.2:
            report += "⚠️ **过拟合严重**（平均差距 > 0.2）\n"
            report += "- 建议: 增强数据增强强度，提升Dropout，增加Weight Decay\n\n"
        elif avg_overfit_gap > 0.1:
            report += "⚡ **轻度过拟合**（平均差距 0.1-0.2）\n"
            report += "- 建议: 保持当前配置，可微调数据增强\n\n"
        else:
            report += "✅ **泛化良好**（平均差距 < 0.1）\n"
            report += "- 建议: 当前配置已优秀，可尝试增加模型容量\n\n"
        
        return report
    
    def _generate_comparison_plots(self, df):
        """生成对比图表"""
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Val F1 对比
        ax = axes[0, 0]
        df_sorted = df.sort_values('best_val_f1', ascending=False)
        colors = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red' 
                  for gap in df_sorted['overfit_gap']]
        ax.bar(range(len(df_sorted)), df_sorted['best_val_f1'], color=colors, alpha=0.7)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Validation F1')
        ax.set_title('Validation F1 Comparison')
        ax.axhline(df_sorted['best_val_f1'].mean(), color='blue', linestyle='--', 
                   label=f'Mean: {df_sorted["best_val_f1"].mean():.4f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Train vs Val F1（过拟合检测）
        ax = axes[0, 1]
        ax.scatter(df['best_train_f1'], df['best_val_f1'], s=100, alpha=0.6)
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Generalization')
        ax.set_xlabel('Train F1')
        ax.set_ylabel('Val F1')
        ax.set_title('Overfitting Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 数据增强效果
        ax = axes[0, 2]
        if 'augmentation_mode' in df.columns:
            aug_groups = df.groupby('augmentation_mode')['best_val_f1'].mean()
            aug_groups.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
            ax.set_ylabel('Mean Val F1')
            ax.set_title('Augmentation Mode Performance')
            ax.tick_params(axis='x', rotation=45)
        
        # 4. Dropout影响
        ax = axes[1, 0]
        dropout_groups = df.groupby('dropout')['best_val_f1'].mean()
        dropout_groups.plot(kind='bar', ax=ax, color='coral', alpha=0.7)
        ax.set_ylabel('Mean Val F1')
        ax.set_title('Dropout Impact')
        ax.tick_params(axis='x', rotation=0)
        
        # 5. 过拟合差距对比
        ax = axes[1, 1]
        df_sorted_gap = df.sort_values('overfit_gap')
        colors_gap = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red' 
                      for gap in df_sorted_gap['overfit_gap']]
        ax.bar(range(len(df_sorted_gap)), df_sorted_gap['overfit_gap'], 
               color=colors_gap, alpha=0.7)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Overfitting Gap (Train F1 - Val F1)')
        ax.set_title('Overfitting Gap Comparison')
        ax.axhline(0.1, color='orange', linestyle='--', label='Warning Threshold')
        ax.axhline(0.2, color='red', linestyle='--', label='Critical Threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 6. 类别F1分布
        ax = axes[1, 2]
        class_f1s = df[['class0_f1', 'class1_f1', 'class2_f1', 'class3_f1']].mean()
        class_f1s.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
                       alpha=0.7)
        ax.set_ylabel('Mean F1')
        ax.set_title('Per-Class Performance')
        ax.set_xticklabels(['Class 0', 'Class 1', 'Class 2', 'Class 3'], rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 对比图表已生成: {plot_path}")
    
    def get_summary(self):
        """获取实验摘要"""
        df = pd.read_csv(self.experiments_file)
        
        print("\n" + "="*70)
        print("实验追踪摘要")
        print("="*70)
        print(f"总实验数: {len(df)}")
        print(f"最高Val F1: {df['best_val_f1'].max():.4f} (实验: {df.loc[df['best_val_f1'].idxmax(), 'experiment_id']})")
        print(f"平均Val F1: {df['best_val_f1'].mean():.4f}")
        print(f"平均过拟合差距: {df['overfit_gap'].mean():.4f}")
        print("="*70)


# ============ 便捷函数 ============

def create_tracker():
    """创建实验追踪器"""
    return ExperimentTracker()


def quick_log(config, results, notes="", tracker=None):
    """快速记录实验"""
    if tracker is None:
        tracker = create_tracker()
    
    return tracker.log_experiment(config, results, notes)


if __name__ == "__main__":
    # 示例使用
    print("实验追踪系统测试\n")
    
    tracker = ExperimentTracker()
    
    # 示例：记录一个baseline实验
    config1 = {
        'description': 'Baseline - 无数据增强',
        'tags': ['baseline', 'no_aug'],
        'time_steps': 12,
        'dropout': 0.1,
        'weight_decay': 1e-4,
        'augmentation_mode': 'none',
        'batch_size': 64,
        'lr': 1e-4,
        'epochs': 100
    }
    
    results1 = {
        'train_size': 4000,
        'val_size': 445,
        'test_size': 1112,
        'best_epoch': 15,
        'best_train_f1': 0.935,
        'best_train_acc': 0.934,
        'best_val_f1': 0.563,
        'best_val_acc': 0.589,
        'final_test_f1': 0.550,
        'final_test_acc': 0.575,
        'class_f1': [0.45, 0.58, 0.60, 0.62],
        'total_params': 129.15e6,
        'trainable_params': 5.52e6,
        'training_time_hours': 2.5
    }
    
    tracker.log_experiment(config1, results1, notes="初始baseline")
    
    # 示例：记录一个带数据增强的实验
    config2 = {
        'description': 'Medium数据增强 + Dropout=0.3',
        'tags': ['augmentation', 'medium'],
        'time_steps': 12,
        'dropout': 0.3,
        'weight_decay': 5e-4,
        'augmentation_mode': 'medium',
        'ts_aug_prob': 0.7,
        'img_aug_prob': 0.5,
        'aug_types': 'noise,scale,shift',
        'batch_size': 64,
        'lr': 1e-4,
        'epochs': 100
    }
    
    results2 = {
        'train_size': 4000,
        'val_size': 445,
        'test_size': 1112,
        'best_epoch': 25,
        'best_train_f1': 0.780,
        'best_train_acc': 0.785,
        'best_val_f1': 0.650,
        'best_val_acc': 0.655,
        'final_test_f1': 0.645,
        'final_test_acc': 0.650,
        'class_f1': [0.55, 0.65, 0.68, 0.70],
        'total_params': 129.15e6,
        'trainable_params': 5.52e6,
        'training_time_hours': 3.0
    }
    
    tracker.log_experiment(config2, results2, notes="数据增强显著改善")
    
    # 生成对比报告
    tracker.compare_experiments()
    
    # 显示摘要
    tracker.get_summary()
    
    print("\n✅ 实验追踪系统测试完成！")

