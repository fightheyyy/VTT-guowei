"""
查看实验结果的便捷脚本
"""

import sys
sys.path.append('../..')

import pandas as pd
import argparse
from experiment_tracker import ExperimentTracker
from pathlib import Path


def show_summary():
    """显示实验摘要"""
    tracker = ExperimentTracker()
    tracker.get_summary()


def show_all():
    """显示所有实验"""
    df = pd.read_csv('experiment_logs/experiments.csv')
    
    print("\n" + "="*100)
    print("所有实验记录")
    print("="*100)
    
    # 选择关键列
    cols = ['experiment_id', 'description', 'augmentation_mode', 
            'dropout', 'best_val_f1', 'final_test_f1', 'overfit_gap']
    
    display_df = df[cols].copy()
    display_df = display_df.sort_values('best_val_f1', ascending=False)
    
    # 格式化
    display_df['best_val_f1'] = display_df['best_val_f1'].apply(lambda x: f"{x:.4f}")
    display_df['final_test_f1'] = display_df['final_test_f1'].apply(lambda x: f"{x:.4f}")
    display_df['overfit_gap'] = display_df['overfit_gap'].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    print("="*100)


def show_top(n=5):
    """显示性能最好的N个实验"""
    df = pd.read_csv('experiment_logs/experiments.csv')
    df = df.sort_values('best_val_f1', ascending=False).head(n)
    
    print(f"\n{'='*100}")
    print(f"Top {n} 实验（按Val F1排序）")
    print("="*100)
    
    for idx, row in enumerate(df.itertuples(), 1):
        medal = "🥇" if idx == 1 else ("🥈" if idx == 2 else ("🥉" if idx == 3 else "  "))
        print(f"\n{medal} 排名 {idx}: {row.experiment_id}")
        print(f"   描述: {row.description}")
        print(f"   增强: {row.augmentation_mode} | Dropout: {row.dropout} | WD: {row.weight_decay}")
        print(f"   Val F1: {row.best_val_f1:.4f} | Test F1: {row.final_test_f1:.4f}")
        print(f"   过拟合: {row.overfit_gap:.4f} (Train F1 - Val F1)")
        print(f"   各类F1: [{row.class0_f1:.3f}, {row.class1_f1:.3f}, {row.class2_f1:.3f}, {row.class3_f1:.3f}]")
    
    print("="*100)


def compare_augmentation():
    """对比不同数据增强模式"""
    df = pd.read_csv('experiment_logs/experiments.csv')
    
    print("\n" + "="*100)
    print("数据增强模式对比")
    print("="*100)
    
    # 按增强模式分组
    grouped = df.groupby('augmentation_mode').agg({
        'best_val_f1': ['mean', 'std', 'max', 'count'],
        'overfit_gap': ['mean', 'std'],
        'final_test_f1': ['mean', 'max']
    }).round(4)
    
    print("\n按增强模式统计:")
    print(grouped.to_string())
    
    # 详细列表
    print("\n\n每种模式的实验:")
    for mode in df['augmentation_mode'].unique():
        mode_df = df[df['augmentation_mode'] == mode].sort_values('best_val_f1', ascending=False)
        print(f"\n{mode.upper()}模式 ({len(mode_df)}个实验):")
        for row in mode_df.itertuples():
            print(f"  - {row.experiment_id}: Val F1={row.best_val_f1:.4f}, Gap={row.overfit_gap:.4f}")
    
    print("="*100)


def show_detail(exp_id):
    """显示单个实验的详细信息"""
    import json
    
    detail_file = f'experiment_logs/{exp_id}_detail.json'
    if not Path(detail_file).exists():
        print(f"❌ 未找到实验详情: {detail_file}")
        return
    
    with open(detail_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*100)
    print(f"实验详情: {exp_id}")
    print("="*100)
    
    print("\n【配置】")
    for key, value in data['config'].items():
        print(f"  {key}: {value}")
    
    print("\n【结果】")
    for key, value in data['results'].items():
        if key != 'class_f1':
            print(f"  {key}: {value}")
        else:
            print(f"  class_f1: {value}")
    
    print(f"\n【备注】")
    print(f"  {data.get('notes', '无')}")
    
    print("="*100)


def generate_comparison():
    """生成对比报告"""
    tracker = ExperimentTracker()
    report_path = tracker.compare_experiments()
    print(f"\n✓ 对比报告已生成: {report_path}")
    print(f"✓ 对比图表已生成: experiment_logs/comparison_plots.png")


def export_for_paper():
    """导出论文所需的表格"""
    df = pd.read_csv('experiment_logs/experiments.csv')
    
    # 选择关键列
    paper_df = df[[
        'description', 'augmentation_mode', 'dropout', 'weight_decay',
        'best_val_f1', 'final_test_f1', 'overfit_gap',
        'class0_f1', 'class1_f1', 'class2_f1', 'class3_f1'
    ]].copy()
    
    paper_df = paper_df.sort_values('best_val_f1', ascending=False)
    
    # 保存为LaTeX表格
    latex_file = 'experiment_logs/paper_table.tex'
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(paper_df.to_latex(index=False, float_format='%.4f'))
    
    print(f"✓ LaTeX表格已导出: {latex_file}")
    
    # 保存为CSV（用于Excel）
    csv_file = 'experiment_logs/paper_table.csv'
    paper_df.to_csv(csv_file, index=False)
    print(f"✓ CSV表格已导出: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查看实验结果')
    parser.add_argument('--summary', action='store_true', help='显示摘要')
    parser.add_argument('--all', action='store_true', help='显示所有实验')
    parser.add_argument('--top', type=int, default=5, help='显示Top N实验')
    parser.add_argument('--augmentation', action='store_true', help='对比数据增强效果')
    parser.add_argument('--detail', type=str, help='显示单个实验详情（实验ID）')
    parser.add_argument('--compare', action='store_true', help='生成对比报告')
    parser.add_argument('--export', action='store_true', help='导出论文表格')
    
    args = parser.parse_args()
    
    # 如果没有参数，显示默认摘要
    if not any(vars(args).values()):
        show_summary()
        print("\n提示：使用 --help 查看更多选项")
    else:
        if args.summary:
            show_summary()
        
        if args.all:
            show_all()
        
        if args.top:
            show_top(args.top)
        
        if args.augmentation:
            compare_augmentation()
        
        if args.detail:
            show_detail(args.detail)
        
        if args.compare:
            generate_comparison()
        
        if args.export:
            export_for_paper()

