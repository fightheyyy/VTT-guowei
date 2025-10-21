"""
灵活的2025年预测脚本
支持不同月份数的输入
"""

import argparse
import numpy as np
from predict_2025 import predict_2025_yield


def main():
    parser = argparse.ArgumentParser(description='灵活的2025年产量预测')
    
    parser.add_argument('--input_months', type=int, default=5,
                        help='输入的月份数（1-11）')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='checkpoints/stage1_timeseries_best.pth',
                        help='阶段1模型路径')
    parser.add_argument('--stage2_checkpoint', type=str,
                        default='checkpoints/stage2_yield_best.pth',
                        help='阶段2模型路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化')
    
    args = parser.parse_args()
    
    # 计算时间步
    lookback = args.input_months * 3
    
    print("=" * 70)
    print("2025年产量预测")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  输入月份: 1-{args.input_months}月")
    print(f"  输入时间步: {lookback}")
    print(f"  预测范围: {args.input_months+1}-12月")
    
    # 定义波段
    band_names = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    n_variates = len(band_names)
    
    # ============ 生成模拟数据（演示用） ============
    print("\n使用模拟数据进行演示...")
    print("⚠️  实际使用时，请替换为真实的2025年观测数据")
    
    np.random.seed(2025)
    input_2025_simulated = np.random.randn(lookback, n_variates) * 1000 + 5000
    
    # ============ 进行预测 ============
    result = predict_2025_yield(
        input_2025_data=input_2025_simulated,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        band_names=band_names,
        device=args.device,
        visualize=args.visualize
    )
    
    print(f"\n" + "=" * 70)
    print("预测结果")
    print("=" * 70)
    print(f"2025年产量预测: {result['yield_prediction']:.2f}")
    print(f"完整全年数据形状: {result['full_year_data'].shape}")
    
    # ============ 使用真实数据的示例代码 ============
    print("\n" + "=" * 70)
    print("使用真实2025年数据的示例代码")
    print("=" * 70)
    print(f"""
# 如果有2025年真实观测数据:
import pandas as pd

# 读取CSV
df_2025 = pd.read_csv('2025_observation.csv')

# 提取前{lookback}个时间步
input_2025_real = []
for band in {band_names}:
    band_cols = [f'{{band}}_{{i:02d}}' for i in range({lookback})]
    band_values = df_2025.loc[0, band_cols].values.astype(np.float32)
    input_2025_real.append(band_values)

input_2025_real = np.array(input_2025_real).T  # [{lookback}, {n_variates}]

# 进行预测
result = predict_2025_yield(
    input_2025_data=input_2025_real,
    stage1_checkpoint='{args.stage1_checkpoint}',
    stage2_checkpoint='{args.stage2_checkpoint}',
    band_names={band_names},
    device='{args.device}',
    visualize=True
)

print(f"2025年产量预测: {{result['yield_prediction']:.2f}}")
    """)


if __name__ == "__main__":
    main()

