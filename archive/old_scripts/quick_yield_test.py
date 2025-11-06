"""
快速测试产量预测
只测试几个关键输入长度，快速验证
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
from train_yield_prediction import experiment_input_length_impact, visualize_results

print("="*70)
print("快速测试：最少需要多少天才能准确预测产量")
print("="*70)

# 配置
train_files = [
    "extract2019_20251010_165007.csv",
    "extract2020_20251010_165007.csv",
    "extract2021_20251010_165007.csv"
]
test_files = [
    "extract2022_20251010_165007.csv"
]

selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n设备: {device}")
print(f"训练数据: {len(train_files)} 年")
print(f"测试数据: {len(test_files)} 年")
print(f"波段数: {len(selected_bands)}")

# 快速测试：只测试几个关键点
print("\n只测试4个关键输入长度:")
print("  - 6步（60天）：极早期")
print("  - 12步（120天）：早期")
print("  - 18步（180天）：中期")
print("  - 30步（300天）：后期")

results = experiment_input_length_impact(
    train_files=train_files,
    test_files=test_files,
    selected_bands=selected_bands,
    input_steps_list=[6, 12, 18, 30],  # 只测试4个点
    model_type='language_only',  # 使用语言模态（更快）
    epochs=30,  # 减少训练轮数
    device=device
)

# 可视化
visualize_results(results, model_type='language_only_quick')

print("\n" + "="*70)
print("快速测试完成！")
print("="*70)
print("\n建议:")
print("  如果结果符合预期，运行完整实验:")
print("    python train_yield_prediction.py")
print("\n  这将测试12个不同输入长度（3-36步）")

