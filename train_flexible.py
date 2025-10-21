"""
灵活配置的两阶段训练
支持不同的输入月份数
"""

import argparse
import os
import torch
from train_two_stage import train_stage1_timeseries, train_stage2_yield


def get_time_config(input_months):
    """
    根据输入月份数计算时间步配置
    
    参数:
        input_months: 输入的月份数（1-11）
    
    返回:
        (lookback, prediction_steps) 元组
    """
    # 每个月约3个时间步（10天一步）
    lookback = input_months * 3
    prediction_steps = 36 - lookback
    
    return lookback, prediction_steps


def main():
    parser = argparse.ArgumentParser(description='灵活配置的TimesCLIP两阶段训练')
    
    # 关键参数：输入月份数
    parser.add_argument('--input_months', type=int, default=5,
                        help='输入的月份数（1-11），默认5表示1-5月')
    
    # 数据配置
    parser.add_argument('--csv_path', type=str, 
                        default='extract2022_20251010_165007.csv',
                        help='训练数据CSV路径')
    parser.add_argument('--bands', type=str, nargs='+',
                        default=['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red'],
                        help='使用的波段列表')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs_stage1', type=int, default=50,
                        help='阶段1训练轮数')
    parser.add_argument('--epochs_stage2', type=int, default=100,
                        help='阶段2训练轮数')
    parser.add_argument('--d_model', type=int, default=256,
                        help='模型隐藏维度')
    
    # 其他配置
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    
    args = parser.parse_args()
    
    # 根据输入月份计算时间步配置
    lookback, prediction_steps = get_time_config(args.input_months)
    
    print("=" * 70)
    print("TimesCLIP 灵活配置训练")
    print("=" * 70)
    print(f"\n配置信息:")
    print(f"  输入月份: 1-{args.input_months}月")
    print(f"  输入时间步: {lookback}")
    print(f"  预测时间步: {prediction_steps} ({args.input_months+1}-12月)")
    print(f"  总时间步: {lookback + prediction_steps} (应=36)")
    print(f"  使用波段: {args.bands}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  模型维度: {args.d_model}")
    print(f"  训练设备: {args.device}")
    
    # 验证配置
    if lookback + prediction_steps != 36:
        raise ValueError(f"时间步数不匹配！lookback({lookback}) + prediction({prediction_steps}) != 36")
    
    if lookback < 3:
        print("\n⚠️  警告: 输入时间步太少(<3)，预测精度可能很差！")
    elif lookback < 9:
        print("\n⚠️  警告: 输入时间步较少(<9)，预测精度可能较低。")
    
    input("\n按Enter键继续训练...")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    # 阶段1: 时间序列补全
    print("\n" + "=" * 70)
    print(f"阶段1: 训练时间序列补全模型 ({lookback}步 → {prediction_steps}步)")
    print("=" * 70)
    
    stage1_model = train_stage1_timeseries(
        csv_path=args.csv_path,
        selected_bands=args.bands,
        lookback=lookback,
        prediction_steps=prediction_steps,
        batch_size=args.batch_size,
        epochs=args.epochs_stage1,
        d_model=args.d_model,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 阶段2: 产量预测
    print("\n" + "=" * 70)
    print("阶段2: 训练产量预测模型 (36步 → 产量)")
    print("=" * 70)
    
    stage2_model = train_stage2_yield(
        csv_path=args.csv_path,
        selected_bands=args.bands,
        target_year='y2022',
        batch_size=args.batch_size * 2,  # 阶段2可以用更大的batch
        epochs=args.epochs_stage2,
        d_model=args.d_model,
        device=args.device,
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n模型已保存到:")
    print(f"  - {args.save_dir}/stage1_timeseries_best.pth")
    print(f"  - {args.save_dir}/stage2_yield_best.pth")
    print(f"\n现在可以使用predict_flexible.py进行预测")
    print(f"  python predict_flexible.py --input_months {args.input_months}")


if __name__ == "__main__":
    main()

