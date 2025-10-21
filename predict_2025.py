"""
2025年产量预测完整流程
步骤1: 使用TimesCLIP将1-5月波段值补全为全年
步骤2: 使用YieldPredictor预测2025年产量
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import TimesCLIP
from models.yield_predictor import YieldPredictor


def load_stage1_model(checkpoint_path, device='cuda'):
    """加载阶段1模型（时间序列补全）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = TimesCLIP(
        time_steps=config['lookback'],
        n_variates=config['n_variates'],
        prediction_steps=config['prediction_steps'],
        patch_length=config['patch_length'],
        stride=config['stride'],
        d_model=config['d_model']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 阶段1模型加载成功 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")
    return model, config


def load_stage2_model(checkpoint_path, device='cuda'):
    """加载阶段2模型（产量预测）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = YieldPredictor(
        n_variates=config['n_variates'],
        time_steps=config['time_steps'],
        d_model=config['d_model']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 阶段2模型加载成功 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}, MAE: {checkpoint['val_mae']:.4f})")
    return model, config


def predict_full_year(input_data, model_stage1, lookback, prediction_steps, device='cuda'):
    """
    步骤1: 使用TimesCLIP补全全年波段值
    
    参数:
        input_data: [lookback, n_variates] 1-5月的波段数据
        model_stage1: TimesCLIP模型
        lookback: 输入长度
        prediction_steps: 预测长度
        device: 设备
    
    返回:
        完整全年数据 [36, n_variates]
    """
    model_stage1.eval()
    
    with torch.no_grad():
        # 转换为tensor并添加batch维度
        x = torch.FloatTensor(input_data).unsqueeze(0).to(device)  # [1, lookback, n_variates]
        
        # 预测6-12月
        y_pred = model_stage1(x, return_loss=False)  # [1, n_variates, prediction_steps]
        y_pred = y_pred.squeeze(0).cpu().numpy().T  # [prediction_steps, n_variates]
    
    # 拼接：前lookback步（1-5月）+ 预测的prediction_steps步（6-12月）
    full_year = np.vstack([input_data, y_pred])  # [lookback + prediction_steps, n_variates]
    
    return full_year


def predict_yield(full_year_data, model_stage2, device='cuda'):
    """
    步骤2: 使用YieldPredictor预测产量
    
    参数:
        full_year_data: [36, n_variates] 完整全年波段数据
        model_stage2: YieldPredictor模型
        device: 设备
    
    返回:
        产量预测值（标量）
    """
    model_stage2.eval()
    
    with torch.no_grad():
        # 转换为tensor并添加batch维度
        x = torch.FloatTensor(full_year_data).unsqueeze(0).to(device)  # [1, 36, n_variates]
        
        # 预测产量
        yield_pred = model_stage2(x)  # [1, 1]
        yield_pred = yield_pred.item()
    
    return yield_pred


def visualize_prediction(input_data, full_year_data, band_names, lookback, save_path='2025_prediction.png'):
    """可视化预测结果"""
    n_variates = len(band_names)
    
    fig, axes = plt.subplots(n_variates, 1, figsize=(15, 3*n_variates))
    if n_variates == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # 输入部分（1-5月）
        ax.plot(range(lookback), input_data[:, i], 'b-', 
                label='已知数据 (1-5月)', linewidth=2, marker='o', markersize=4)
        
        # 预测部分（6-12月）
        ax.plot(range(lookback, len(full_year_data)), full_year_data[lookback:, i], 'r--', 
                label='预测数据 (6-12月)', linewidth=2, marker='s', markersize=4)
        
        # 分隔线
        ax.axvline(x=lookback, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(lookback, ax.get_ylim()[1]*0.95, '预测起点', 
                ha='center', va='top', fontsize=10, color='gray')
        
        ax.set_title(f'{band_names[i]} 波段 - 2025年预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步（每步10天）', fontsize=12)
        ax.set_ylabel('波段值', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存: {save_path}")
    plt.close()


def predict_2025_yield(
    input_2025_data,
    stage1_checkpoint='checkpoints/stage1_timeseries_best.pth',
    stage2_checkpoint='checkpoints/stage2_yield_best.pth',
    band_names=None,
    device='cuda',
    visualize=True
):
    """
    完整的2025年产量预测流程
    
    参数:
        input_2025_data: [lookback, n_variates] 2025年1-5月的波段数据
        stage1_checkpoint: 阶段1模型路径
        stage2_checkpoint: 阶段2模型路径
        band_names: 波段名称列表
        device: 设备
        visualize: 是否可视化
    
    返回:
        {
            'full_year_data': 完整全年数据,
            'yield_prediction': 产量预测值
        }
    """
    print("=" * 70)
    print("2025年产量预测流程")
    print("=" * 70)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    model_stage1, config1 = load_stage1_model(stage1_checkpoint, device)
    model_stage2, config2 = load_stage2_model(stage2_checkpoint, device)
    
    lookback = config1['lookback']
    prediction_steps = config1['prediction_steps']
    
    # 验证输入数据
    print(f"\n[2/4] 验证输入数据...")
    print(f"  - 输入形状: {input_2025_data.shape}")
    print(f"  - 期望形状: [{lookback}, {config1['n_variates']}]")
    
    if input_2025_data.shape[0] != lookback:
        raise ValueError(f"输入时间步数应为{lookback}，但得到{input_2025_data.shape[0]}")
    if input_2025_data.shape[1] != config1['n_variates']:
        raise ValueError(f"波段数应为{config1['n_variates']}，但得到{input_2025_data.shape[1]}")
    
    print("  ✓ 输入数据验证通过")
    
    # 步骤1: 补全全年波段值
    print(f"\n[3/4] 预测全年波段值...")
    print(f"  - 已知: 前{lookback}个时间步 (1-5月)")
    print(f"  - 预测: 后{prediction_steps}个时间步 (6-12月)")
    
    full_year_data = predict_full_year(
        input_2025_data, model_stage1, lookback, prediction_steps, device
    )
    
    print(f"  ✓ 全年数据形状: {full_year_data.shape}")
    
    # 步骤2: 预测产量
    print(f"\n[4/4] 预测2025年产量...")
    yield_prediction = predict_yield(full_year_data, model_stage2, device)
    
    print(f"\n" + "=" * 70)
    print(f"预测结果")
    print("=" * 70)
    print(f"2025年预测产量: {yield_prediction:.2f}")
    print("=" * 70)
    
    # 可视化
    if visualize and band_names is not None:
        print("\n生成可视化...")
        visualize_prediction(
            input_2025_data, full_year_data, band_names, lookback
        )
    
    return {
        'full_year_data': full_year_data,
        'yield_prediction': yield_prediction
    }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    示例：使用模拟的2025年1-5月数据进行预测
    实际使用时，需要替换为真实的2025年观测数据
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 定义波段
    band_names = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    n_variates = len(band_names)
    lookback = 18  # 1-5月约18个时间步
    
    # ============ 方法1: 使用模拟数据（演示） ============
    print("方法1: 使用模拟数据进行演示")
    print("-" * 70)
    
    # 生成模拟的2025年1-5月数据
    # 实际使用时，应从真实观测数据中读取
    np.random.seed(2025)
    input_2025_simulated = np.random.randn(lookback, n_variates) * 1000 + 5000
    
    result = predict_2025_yield(
        input_2025_data=input_2025_simulated,
        band_names=band_names,
        device=device,
        visualize=True
    )
    
    print(f"\n完整全年数据形状: {result['full_year_data'].shape}")
    print(f"2025年产量预测: {result['yield_prediction']:.2f}")
    
    # ============ 方法2: 从CSV读取真实数据（实际使用） ============
    print("\n\n方法2: 从CSV读取2025年真实数据")
    print("-" * 70)
    print("说明: 当有2025年1-5月真实观测数据时，使用此方法")
    print("""
# 示例代码：
csv_path = '2025_data.csv'  # 包含2025年1-5月观测数据的CSV文件
df = pd.read_csv(csv_path)

# 提取第一个样本的前18步数据
input_2025_real = []
for band in band_names:
    band_cols = [f'{band}_{i:02d}' for i in range(lookback)]
    band_values = df.loc[0, band_cols].values.astype(np.float32)
    input_2025_real.append(band_values)

input_2025_real = np.array(input_2025_real).T  # [lookback, n_variates]

# 进行预测
result = predict_2025_yield(
    input_2025_data=input_2025_real,
    band_names=band_names,
    device=device
)

print(f"2025年产量预测: {result['yield_prediction']:.2f}")
    """)
    
    print("\n" + "=" * 70)
    print("预测流程说明")
    print("=" * 70)
    print("""
完整预测流程:
1. 准备2025年1-5月的波段观测数据（7个波段 × 18个时间步）
2. 使用TimesCLIP模型补全6-12月的波段值
3. 获得完整全年36个时间步的波段数据
4. 使用YieldPredictor模型预测2025年产量

输入要求:
- 形状: [18, 7] 即18个时间步，7个波段
- 波段顺序: NIR, RVI, SWIR1, blue, evi, ndvi, red
- 数据来源: 遥感卫星观测数据

输出结果:
- full_year_data: [36, 7] 完整全年波段数据
- yield_prediction: 标量，预测的2025年产量值
    """)

