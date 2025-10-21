"""
测试数据加载是否正常工作
"""

import torch
from data_loader import create_dataloaders

print("=" * 70)
print("测试数据加载器")
print("=" * 70)

# 配置
csv_path = "extract2022_20251010_165007.csv"

# 测试1: 使用部分波段
print("\n[测试1] 使用7个常用波段")
print("-" * 70)
selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']

try:
    train_loader, test_loader, n_variates = create_dataloaders(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=24,
        prediction_steps=12,
        batch_size=8,
        test_size=0.2,
        random_state=42
    )
    print(f"\n✓ 数据加载成功！")
    print(f"  - 波段数: {n_variates}")
    print(f"  - 训练批次数: {len(train_loader)}")
    print(f"  - 测试批次数: {len(test_loader)}")
    
    # 测试获取一个批次
    for x, y in train_loader:
        print(f"\n批次数据形状:")
        print(f"  - 输入 X: {x.shape} [batch_size, lookback, n_variates]")
        print(f"  - 目标 Y: {y.shape} [batch_size, n_variates, prediction_steps]")
        print(f"\n数据统计:")
        print(f"  - 输入值范围: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  - 目标值范围: [{y.min():.2f}, {y.max():.2f}]")
        break
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 使用所有波段
print("\n" + "=" * 70)
print("[测试2] 使用全部14个波段")
print("-" * 70)

try:
    train_loader, test_loader, n_variates = create_dataloaders(
        csv_path=csv_path,
        selected_bands=None,  # None表示使用所有波段
        lookback=24,
        prediction_steps=12,
        batch_size=8
    )
    print(f"\n✓ 数据加载成功！")
    print(f"  - 波段数: {n_variates}")
    
except Exception as e:
    print(f"✗ 错误: {e}")

# 测试3: 不同的时间窗口
print("\n" + "=" * 70)
print("[测试3] 测试不同的时间窗口配置")
print("-" * 70)

configs = [
    {"lookback": 20, "prediction_steps": 16},
    {"lookback": 28, "prediction_steps": 8},
    {"lookback": 18, "prediction_steps": 18},
]

for i, config in enumerate(configs, 1):
    try:
        train_loader, test_loader, n_variates = create_dataloaders(
            csv_path=csv_path,
            selected_bands=selected_bands,
            lookback=config["lookback"],
            prediction_steps=config["prediction_steps"],
            batch_size=4
        )
        print(f"\n配置{i}: lookback={config['lookback']}, prediction={config['prediction_steps']} ✓")
        
        # 验证总长度不超过36
        total_steps = config['lookback'] + config['prediction_steps']
        if total_steps <= 36:
            print(f"  ✓ 总时间步={total_steps}，在有效范围内")
        else:
            print(f"  ⚠ 总时间步={total_steps}，超过36步")
            
    except Exception as e:
        print(f"\n配置{i}: ✗ 错误: {e}")

print("\n" + "=" * 70)
print("数据加载测试完成！")
print("=" * 70)

print("\n提示:")
print("- 如果所有测试通过，可以运行 'python train.py' 开始训练")
print("- 建议的配置: lookback=24, prediction_steps=12")
print("- 总时间步（lookback + prediction_steps）应 ≤ 36")

