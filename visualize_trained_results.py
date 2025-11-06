"""
快速可视化已训练的结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 读取结果
with open('experiments/yield_prediction/results/results.json', 'r') as f:
    results = json.load(f)

# 提取数据
input_steps = sorted([int(k) for k in results.keys()])
days = [results[s]['days'] for s in input_steps]
rmse_values = [results[s]['rmse'] for s in input_steps]
r2_values = [results[s]['r2'] for s in input_steps]
mae_values = [results[s]['mae'] for s in input_steps]
mape_values = [results[s]['mape'] for s in input_steps]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# RMSE
ax = axes[0, 0]
ax.plot(days, rmse_values, 'o-', linewidth=2, markersize=8, color='#e74c3c')
best_idx = np.argmin(rmse_values)
ax.plot(days[best_idx], rmse_values[best_idx], 'r*', markersize=20, label=f'最佳: {days[best_idx]}天')
ax.set_xlabel('输入天数', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('RMSE vs 输入天数')
ax.legend()
ax.grid(True, alpha=0.3)

# R²
ax = axes[0, 1]
ax.plot(days, r2_values, 'o-', linewidth=2, markersize=8, color='#2ecc71')
ax.set_xlabel('输入天数', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('R² vs 输入天数')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# MAE
ax = axes[1, 0]
ax.plot(days, mae_values, 'o-', linewidth=2, markersize=8, color='#3498db')
ax.set_xlabel('输入天数', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('MAE vs 输入天数')
ax.grid(True, alpha=0.3)

# MAPE
ax = axes[1, 1]
ax.plot(days, mape_values, 'o-', linewidth=2, markersize=8, color='#9b59b6')
ax.set_xlabel('输入天数', fontsize=12)
ax.set_ylabel('MAPE (%)', fontsize=12)
ax.set_title('MAPE vs 输入天数')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/yield_prediction/results/analysis.png', dpi=300, bbox_inches='tight')
print("\n图表已保存: experiments/yield_prediction/results/analysis.png")

# 打印最佳结果
print(f"\n{'='*70}")
print("最佳结果:")
print(f"{'='*70}")
print(f"最低RMSE: {rmse_values[best_idx]:.4f} (在{days[best_idx]}天时)")
print(f"对应R²: {r2_values[best_idx]:.4f}")
print(f"对应MAE: {mae_values[best_idx]:.4f}")
print(f"对应MAPE: {mape_values[best_idx]:.2f}%")

print(f"\n{'='*70}")
print("所有结果:")
print(f"{'='*70}")
for i, step in enumerate(input_steps):
    print(f"{days[i]:3d}天 ({step:2d}步): RMSE={rmse_values[i]:.4f}, R²={r2_values[i]:.4f}, MAE={mae_values[i]:.4f}, MAPE={mape_values[i]:.2f}%")

plt.close()

