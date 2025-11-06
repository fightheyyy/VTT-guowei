"""
测试可变长度预测功能
"""

import torch
from models.timesclip_variable_length import TimesCLIPVariableLength
from data_loader_variable_length import create_variable_length_dataloaders
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


def test_model_with_different_lengths():
    """测试模型在不同输入长度下的表现"""
    
    print("=" * 70)
    print("测试可变长度预测模型")
    print("=" * 70)
    
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
    
    # 初始化模型
    model = TimesCLIPVariableLength(
        max_time_steps=36,
        n_variates=7,
        patch_length=6,
        stride=3,
        d_model=256
    ).to(device)
    
    # 测试不同输入长度
    test_lengths = [3, 6, 12, 18, 24, 30]
    
    for test_len in test_lengths:
        print(f"\n{'='*70}")
        print(f"测试：前{test_len}个月 → 预测后{36 - test_len}个月")
        print(f"{'='*70}")
        
        # 加载测试数据
        _, test_loader, _ = create_variable_length_dataloaders(
            train_csv_paths=train_files,
            test_csv_paths=test_files,
            selected_bands=selected_bands,
            max_time_steps=36,
            min_input_length=3,
            max_input_length=30,
            test_input_length=test_len,
            batch_size=4
        )
        
        # 获取一个batch
        x, y_true, input_lengths = next(iter(test_loader))
        x = x.to(device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            y_pred = model(x[:1, :test_len, :], input_length=test_len)
        
        print(f"输入形状: {x[:1, :test_len, :].shape}")
        print(f"预测形状: {y_pred.shape}")
        print(f"真实形状: {y_true[:1, :36-test_len, :].shape}")
        
        # 可视化第一个样本的第一个变量
        x_np = x[0, :test_len, 0].cpu().numpy()
        y_true_np = y_true[0, :36-test_len, 0].cpu().numpy()
        y_pred_np = y_pred[0, :, 0].cpu().numpy()
        
        plt.figure(figsize=(12, 4))
        
        # 绘制已知部分
        plt.plot(range(test_len), x_np, 'b-', label='已知数据', linewidth=2)
        
        # 绘制真实未来部分
        plt.plot(range(test_len, 36), y_true_np, 'g-', label='真实未来', linewidth=2)
        
        # 绘制预测未来部分
        plt.plot(range(test_len, 36), y_pred_np, 'r--', label='预测未来', linewidth=2)
        
        # 标记分界线
        plt.axvline(x=test_len, color='gray', linestyle=':', label='预测起点')
        
        plt.xlabel('时间（月）')
        plt.ylabel('归一化值')
        plt.title(f'前{test_len}个月预测后{36-test_len}个月（{selected_bands[0]}波段）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'predictions/variable_length_{test_len}months.png', dpi=150)
        plt.close()
        
        print(f"可视化保存到: predictions/variable_length_{test_len}months.png")


def demo_early_prediction():
    """演示早期预测的实用场景"""
    
    print("\n" + "=" * 70)
    print("早期预测演示场景")
    print("=" * 70)
    
    scenarios = [
        {
            'name': '极早期预测（播种后3个月）',
            'input_months': 3,
            'description': '刚完成播种和早期生长，数据最少但预测时间最长'
        },
        {
            'name': '早期预测（生长期6个月）',
            'input_months': 6,
            'description': '作物进入生长期，有了初步的生长数据'
        },
        {
            'name': '中期预测（生长期12个月）',
            'input_months': 12,
            'description': '已有完整的生长周期数据，预测准确度提升'
        },
        {
            'name': '后期预测（收获前18个月）',
            'input_months': 18,
            'description': '接近收获期，数据最完整，预测最准确'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'─'*70}")
        print(f"场景：{scenario['name']}")
        print(f"说明：{scenario['description']}")
        print(f"输入：前{scenario['input_months']}个月")
        print(f"预测：后{36 - scenario['input_months']}个月")
        print(f"优势：{'越早预测越有价值' if scenario['input_months'] < 12 else '准确度高'}")


if __name__ == "__main__":
    import os
    os.makedirs('predictions', exist_ok=True)
    
    # 演示场景
    demo_early_prediction()
    
    # 测试模型
    print("\n\n")
    test_model_with_different_lengths()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("\n使用说明：")
    print("1. 先运行训练: python train_variable_length.py")
    print("2. 训练完成后加载模型进行预测")
    print("3. 可以使用任意前N个月（3-30个月）进行预测")
    print("\n实际应用场景：")
    print("  - 前3个月：极早期产量预估，用于市场规划")
    print("  - 前6个月：早期预警，调整种植策略")
    print("  - 前12个月：中期预测，优化资源配置")
    print("  - 前18个月：准确预测，制定收获计划")

