"""
自动化消融实验脚本
依次执行：双模态训练 -> 语言模态训练 -> 结果对比
"""

import os
import sys
import time
import subprocess
from datetime import datetime


def print_header(text, char='='):
    """打印标题"""
    print(f"\n{char * 70}")
    print(text)
    print(f"{char * 70}\n")


def run_command(cmd, log_file, description):
    """执行命令并记录日志"""
    print(f"\n{description}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志保存: {log_file}")
    print(f"命令: {cmd}")
    print("-" * 70)
    
    start_time = time.time()
    
    # 执行命令，实时显示输出
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            shell=True
        )
        
        # 实时打印输出
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
        exit_code = process.returncode
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"耗时: {hours}小时 {minutes}分钟 {seconds}秒")
    
    if exit_code != 0:
        print(f"\n❌ 错误: 命令执行失败 (退出码: {exit_code})")
        print(f"请查看日志: {log_file}")
        return False
    
    print(f"✓ 成功完成")
    return True


def check_python():
    """检查Python环境"""
    print("检查Python环境...")
    result = subprocess.run(['python', '--version'], capture_output=True, text=True)
    print(f"  Python版本: {result.stdout.strip()}")
    
    # 检查关键包
    try:
        import torch
        import transformers
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  Transformers版本: {transformers.__version__}")
        print("✓ 环境检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        return False


def main():
    print_header("消融实验：对比双模态 vs 语言模态")
    
    # 记录实验开始时间
    experiment_start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"时间戳: {timestamp}")
    
    # 创建日志目录
    log_dir = 'experiment_logs'
    os.makedirs(log_dir, exist_ok=True)
    print(f"日志目录: {log_dir}/")
    
    # 检查环境
    if not check_python():
        print("\n请先安装依赖: pip install -r requirements.txt")
        sys.exit(1)
    
    # 步骤1: 训练双模态版本
    print_header("[1/3] 训练双模态版本", '-')
    print("预计时间: 8-10 小时")
    print("模型: TimesCLIP (视觉 + 语言 + 对比学习)")
    
    step1_log = os.path.join(log_dir, f'step1_both_modal_{timestamp}.log')
    if not run_command(
        'python train_multiyear_mirror.py',
        step1_log,
        '开始训练双模态模型...'
    ):
        print("\n实验中止")
        sys.exit(1)
    
    print(f"\n✓ 双模态模型训练完成")
    print(f"  模型保存: checkpoints/stage1_timeseries_best.pth")
    
    # 步骤2: 训练语言模态版本
    print_header("[2/3] 训练语言模态版本", '-')
    print("预计时间: 2-3 小时")
    print("模型: TimesCLIPLanguageOnly (仅语言模态)")
    
    step2_log = os.path.join(log_dir, f'step2_language_only_{timestamp}.log')
    if not run_command(
        'python train_language_only.py',
        step2_log,
        '开始训练语言模态模型...'
    ):
        print("\n实验中止")
        sys.exit(1)
    
    print(f"\n✓ 语言模态模型训练完成")
    print(f"  模型保存: checkpoints/stage1_language_only_best.pth")
    
    # 步骤3: 对比结果
    print_header("[3/3] 对比两个版本的性能", '-')
    
    step3_log = os.path.join(log_dir, f'step3_comparison_{timestamp}.log')
    if not run_command(
        'python compare_results.py',
        step3_log,
        '开始对比分析...'
    ):
        print("\n⚠ 警告: 对比脚本执行失败")
    
    # 实验总结
    experiment_elapsed = time.time() - experiment_start_time
    total_hours = int(experiment_elapsed // 3600)
    total_minutes = int((experiment_elapsed % 3600) // 60)
    
    print_header("✓ 实验全部完成！")
    
    print(f"实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_hours}小时 {total_minutes}分钟")
    print()
    
    print("结果文件:")
    print("  - checkpoints/stage1_timeseries_best.pth (双模态)")
    print("  - checkpoints/stage1_language_only_best.pth (语言模态)")
    print("  - comparison_results.json (对比结果)")
    print()
    
    print("日志文件:")
    print(f"  - {step1_log}")
    print(f"  - {step2_log}")
    print(f"  - {step3_log}")
    print()
    
    print("查看训练曲线:")
    print("  tensorboard --logdir=logs")
    print()
    
    # 读取对比结果
    if os.path.exists('comparison_results.json'):
        print("=" * 70)
        print("对比结果摘要:")
        print("=" * 70)
        try:
            import json
            with open('comparison_results.json', 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if 'stage1_both' in results and 'stage1_language' in results:
                both = results['stage1_both']
                lang = results['stage1_language']
                
                if both and lang:
                    rmse_diff = (lang['rmse'] - both['rmse']) / both['rmse'] * 100
                    
                    print(f"\n阶段1 - 时间序列补全:")
                    print(f"  双模态 RMSE:    {both['rmse']:.4f}")
                    print(f"  语言模态 RMSE:  {lang['rmse']:.4f}")
                    print(f"  差异:           {rmse_diff:+.1f}%")
                    print(f"  性能保留率:     {100 - abs(rmse_diff):.1f}%")
                    
                    if abs(rmse_diff) < 5:
                        print(f"\n  结论: ⭐ 语言模态性能相当（差异<5%），建议去掉视觉模态")
                    elif abs(rmse_diff) < 10:
                        print(f"\n  结论: ⚠ 语言模态性能略有差异（5-10%），需权衡")
                    else:
                        print(f"\n  结论: ❌ 视觉模态显著提升性能（>10%），建议保留")
        except Exception as e:
            print(f"无法解析结果文件: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

