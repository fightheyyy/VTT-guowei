"""
测试实验追踪系统
验证系统是否正常工作
"""

import sys
sys.path.append('../..')

from experiment_tracker import ExperimentTracker
import os
import shutil


def test_basic_logging():
    """测试基本记录功能"""
    print("\n" + "="*70)
    print("测试1: 基本实验记录")
    print("="*70)
    
    tracker = ExperimentTracker(log_dir="test_logs")
    
    # 测试配置
    config = {
        'description': '测试实验 - Baseline',
        'tags': ['test', 'baseline'],
        'time_steps': 12,
        'dropout': 0.1,
        'weight_decay': 1e-4,
        'augmentation_mode': 'none',
        'batch_size': 64,
        'lr': 1e-4,
        'epochs': 100
    }
    
    # 测试结果
    results = {
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
    
    exp_id = tracker.log_experiment(config, results, notes="测试实验")
    
    assert exp_id is not None
    print("✓ 基本记录测试通过")
    
    return tracker


def test_multiple_experiments(tracker):
    """测试记录多个实验"""
    print("\n" + "="*70)
    print("测试2: 记录多个实验")
    print("="*70)
    
    experiments = [
        {
            'config': {
                'description': '测试 - Light增强',
                'tags': ['test', 'light'],
                'augmentation_mode': 'light',
                'dropout': 0.2,
                'weight_decay': 3e-4,
                'time_steps': 12,
                'batch_size': 64,
                'lr': 1e-4,
                'epochs': 100
            },
            'results': {
                'train_size': 4000,
                'val_size': 445,
                'test_size': 1112,
                'best_epoch': 18,
                'best_train_f1': 0.850,
                'best_train_acc': 0.855,
                'best_val_f1': 0.605,
                'best_val_acc': 0.610,
                'final_test_f1': 0.595,
                'final_test_acc': 0.600,
                'class_f1': [0.52, 0.62, 0.64, 0.64],
                'total_params': 129.15e6,
                'trainable_params': 5.52e6,
                'training_time_hours': 2.8
            }
        },
        {
            'config': {
                'description': '测试 - Medium增强',
                'tags': ['test', 'medium'],
                'augmentation_mode': 'medium',
                'dropout': 0.3,
                'weight_decay': 5e-4,
                'time_steps': 12,
                'batch_size': 64,
                'lr': 1e-4,
                'epochs': 100
            },
            'results': {
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
        },
        {
            'config': {
                'description': '测试 - Heavy增强',
                'tags': ['test', 'heavy'],
                'augmentation_mode': 'heavy',
                'dropout': 0.3,
                'weight_decay': 5e-4,
                'time_steps': 12,
                'batch_size': 64,
                'lr': 1e-4,
                'epochs': 100
            },
            'results': {
                'train_size': 4000,
                'val_size': 445,
                'test_size': 1112,
                'best_epoch': 30,
                'best_train_f1': 0.720,
                'best_train_acc': 0.725,
                'best_val_f1': 0.680,
                'best_val_acc': 0.685,
                'final_test_f1': 0.675,
                'final_test_acc': 0.680,
                'class_f1': [0.60, 0.68, 0.72, 0.75],
                'total_params': 129.15e6,
                'trainable_params': 5.52e6,
                'training_time_hours': 3.5
            }
        }
    ]
    
    for i, exp in enumerate(experiments, 1):
        tracker.log_experiment(exp['config'], exp['results'], 
                              notes=f"测试实验{i}")
    
    print(f"✓ 成功记录{len(experiments)}个实验")


def test_comparison(tracker):
    """测试对比功能"""
    print("\n" + "="*70)
    print("测试3: 生成对比报告")
    print("="*70)
    
    report_path = tracker.compare_experiments(tags=['test'])
    
    assert os.path.exists(report_path)
    assert os.path.exists('test_logs/comparison_plots.png')
    
    print("✓ 对比报告生成测试通过")


def test_summary(tracker):
    """测试摘要功能"""
    print("\n" + "="*70)
    print("测试4: 显示摘要")
    print("="*70)
    
    tracker.get_summary()
    
    print("✓ 摘要显示测试通过")


def cleanup():
    """清理测试文件"""
    print("\n" + "="*70)
    print("清理测试文件")
    print("="*70)
    
    if os.path.exists('test_logs'):
        shutil.rmtree('test_logs')
        print("✓ 测试文件已清理")


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "实验追踪系统测试" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    try:
        # 测试1: 基本记录
        tracker = test_basic_logging()
        
        # 测试2: 多实验记录
        test_multiple_experiments(tracker)
        
        # 测试3: 对比报告
        test_comparison(tracker)
        
        # 测试4: 摘要显示
        test_summary(tracker)
        
        print("\n" + "="*70)
        print("✅ 所有测试通过！实验追踪系统运行正常")
        print("="*70)
        print("\n建议:")
        print("  1. 查看生成的测试报告: test_logs/comparison_report.md")
        print("  2. 查看生成的对比图表: test_logs/comparison_plots.png")
        print("  3. 开始正式实验: python train_12steps_dual_cached.py")
        print("="*70)
        
        # 询问是否清理
        response = input("\n是否清理测试文件？(y/n): ")
        if response.lower() == 'y':
            cleanup()
        else:
            print("✓ 测试文件保留在 test_logs/ 目录")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理
        if os.path.exists('test_logs'):
            shutil.rmtree('test_logs')

