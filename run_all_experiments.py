"""
运行所有实验的主脚本
"""

import os
import subprocess
import time
from datetime import datetime


def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def run_experiment(script_path, name, args=""):
    """运行单个实验"""
    print_header(f"开始实验: {name}")
    print(f"脚本: {script_path}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        cmd = f"python {script_path} {args}"
        print(f"命令: {cmd}\n")
        
        result = subprocess.run(cmd, shell=True, check=True, 
                               capture_output=False, text=True)
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print_header(f"✓ {name} 完成")
        print(f"耗时: {hours}小时 {minutes}分钟")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_header(f"✗ {name} 失败")
        print(f"错误: {e}")
        return False


def main():
    print_header("VTT 实验套件")
    print("将依次运行4个实验:")
    print("  1. 产量预测（最短有效天数）")
    print("  2. 补全对比（补全 vs 不补全）")
    print("  3. 两阶段训练")
    print("  4. 可变长度预测")
    
    print("\n预计总时间: 10-15小时")
    
    response = input("\n是否继续？(y/n): ")
    if response.lower() != 'y':
        print("取消运行")
        return
    
    # 记录开始时间
    total_start = time.time()
    
    results = {}
    
    # 实验1: 产量预测（快速模式）
    results['yield_prediction'] = run_experiment(
        "experiments/yield_prediction/train.py",
        "实验1: 产量预测",
        "--quick"  # 快速模式
    )
    
    # 实验2: 补全对比
    if results['yield_prediction']:
        results['completion_comparison'] = run_experiment(
            "experiments/completion_comparison/train.py",
            "实验2: 补全对比"
        )
    
    # 实验3: 两阶段训练
    if results.get('completion_comparison', False):
        results['two_stage'] = run_experiment(
            "experiments/two_stage/train.py",
            "实验4: 两阶段训练"
        )
    
    # 实验4: 可变长度（可选，较复杂）
    print_header("实验3: 可变长度预测")
    print("这个实验较复杂且耗时，是否运行？")
    response = input("(y/n): ")
    
    if response.lower() == 'y':
        results['variable_length'] = run_experiment(
            "experiments/variable_length/train.py",
            "实验3: 可变长度预测"
        )
    else:
        results['variable_length'] = None
        print("跳过可变长度实验")
    
    # 总结
    total_elapsed = time.time() - total_start
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)
    
    print_header("所有实验完成")
    print(f"总耗时: {total_hours}小时 {total_minutes}分钟\n")
    
    print("实验结果:")
    for exp_name, success in results.items():
        if success is None:
            status = "⊘ 跳过"
        elif success:
            status = "✓ 成功"
        else:
            status = "✗ 失败"
        print(f"  {status} {exp_name}")
    
    print("\n查看结果:")
    print("  - experiments/yield_prediction/results/")
    print("  - experiments/completion_comparison/results/")
    print("  - experiments/two_stage/results/")
    if results.get('variable_length'):
        print("  - experiments/variable_length/results/")
    
    print("\n查看训练曲线:")
    print("  tensorboard --logdir=experiments/yield_prediction/logs")


if __name__ == "__main__":
    main()

