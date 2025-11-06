"""
诊断训练结果
"""

import pandas as pd
import numpy as np

# 1. 检查数据分布
print("="*70)
print("数据诊断")
print("="*70)

for year in [2019, 2020, 2021, 2022]:
    df = pd.read_csv(f'data/extract{year}_20251010_165007.csv')
    y_col = f'y{year}'
    print(f"\n{year}年产量统计:")
    print(f"  均值: {df[y_col].mean():.4f}")
    print(f"  标准差: {df[y_col].std():.4f}")
    print(f"  最小值: {df[y_col].min():.4f}")
    print(f"  最大值: {df[y_col].max():.4f}")
    print(f"  变异系数: {df[y_col].std() / df[y_col].mean():.4f}")
    print(f"  零值数量: {(df[y_col] == 0).sum()}")

# 2. 基线对比
print("\n" + "="*70)
print("基线模型性能（简单使用均值预测）")
print("="*70)

# 训练集均值
train_dfs = []
for year in [2019, 2020, 2021]:
    df = pd.read_csv(f'data/extract{year}_20251010_165007.csv')
    train_dfs.append(df[f'y{year}'].values)
train_yields = np.concatenate(train_dfs)
train_mean = train_yields.mean()

# 测试集
test_df = pd.read_csv('data/extract2022_20251010_165007.csv')
test_yields = test_df['y2022'].values

# 基线预测（使用训练集均值）
baseline_preds = np.ones_like(test_yields) * train_mean
baseline_errors = test_yields - baseline_preds
baseline_mse = np.mean(baseline_errors ** 2)
baseline_rmse = np.sqrt(baseline_mse)
baseline_mae = np.mean(np.abs(baseline_errors))

# R² = 1 - SS_res / SS_tot
ss_tot = np.sum((test_yields - test_yields.mean()) ** 2)
ss_res = np.sum(baseline_errors ** 2)
baseline_r2 = 1 - ss_res / ss_tot

print(f"\n训练集均值: {train_mean:.4f}")
print(f"测试集均值: {test_yields.mean():.4f}")
print(f"测试集标准差: {test_yields.std():.4f}")
print(f"\n基线性能（预测=训练均值）:")
print(f"  RMSE: {baseline_rmse:.4f}")
print(f"  MAE:  {baseline_mae:.4f}")
print(f"  R2:   {baseline_r2:.4f}")

print("\n" + "="*70)
print("对比模型性能")
print("="*70)
print("\n如果模型的RMSE > {:.4f}，说明模型比基线还差！".format(baseline_rmse))
print("如果模型的R2 < 0，说明模型比使用均值还差！")

# 3. 数据变异性分析
print("\n" + "="*70)
print("数据变异性分析")
print("="*70)

print(f"\n测试集产量变化范围: {test_yields.max() - test_yields.min():.4f}")
print(f"相对变化: {(test_yields.max() - test_yields.min()) / test_yields.mean() * 100:.2f}%")

# 检查序列数据的变异性
selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
df = pd.read_csv('data/extract2022_20251010_165007.csv')

print(f"\n遥感数据变异性:")
for band in selected_bands:
    cols = [f"{band}_{i:02d}" for i in range(36)]
    values = df[cols].values.flatten()
    print(f"  {band:6s}: 均值={values.mean():8.3f}, 标准差={values.std():8.3f}, 变异系数={values.std()/np.abs(values.mean()):.3f}")

print("\n" + "="*70)
print("结论")
print("="*70)
print("\n1. 产量数据的变异性很小（标准差/均值约5%）")
print("2. 这使得预测任务非常困难")
print("3. 需要模型能够捕捉到很细微的差异")
print("4. 建议:")
print("   - 增加训练轮数")
print("   - 调整学习率")
print("   - 使用更深的网络")
print("   - 尝试数据增强")

