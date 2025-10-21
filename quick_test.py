"""
快速测试脚本 - 验证TimesCLIP模型基本功能
"""

import torch
from models import TimesCLIP

print("=" * 50)
print("TimesCLIP 模型快速测试")
print("=" * 50)

# 测试1: 模型初始化
print("\n[1/4] 测试模型初始化...")
try:
    model = TimesCLIP(
        time_steps=96,
        n_variates=7,
        prediction_steps=96,
        patch_length=16,
        stride=8,
        d_model=256,  # 使用较小的维度以加快测试速度
        n_heads=4
    )
    print("✓ 模型初始化成功")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    exit(1)

# 测试2: 参数统计
print("\n[2/4] 统计模型参数...")
try:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 总参数量: {total_params:,}")
    print(f"✓ 可训练参数: {trainable_params:,}")
except Exception as e:
    print(f"✗ 参数统计失败: {e}")

# 测试3: 前向传播（推理模式）
print("\n[3/4] 测试前向传播（推理模式）...")
try:
    model.eval()
    x = torch.randn(2, 96, 7)  # 小批量测试
    with torch.no_grad():
        y_pred = model(x, return_loss=False)
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {y_pred.shape}")
    assert y_pred.shape == (2, 7, 96), "输出形状不正确"
    print("✓ 前向传播成功")
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    exit(1)

# 测试4: 前向传播（训练模式，包含损失）
print("\n[4/4] 测试前向传播（训练模式）...")
try:
    model.train()
    x = torch.randn(2, 96, 7)
    y_pred, contrastive_loss = model(x, return_loss=True)
    print(f"✓ 预测形状: {y_pred.shape}")
    print(f"✓ 对比损失: {contrastive_loss.item():.4f}")
    
    # 测试损失计算
    y_true = torch.randn(2, 7, 96)
    total_loss, loss_dict = model.compute_loss(
        y_pred, y_true, contrastive_loss,
        lambda_gen=1.0, lambda_align=0.1
    )
    print(f"✓ 总损失: {loss_dict['total_loss']:.4f}")
    print(f"✓ MSE损失: {loss_dict['mse_loss']:.4f}")
    print("✓ 训练模式测试成功")
except Exception as e:
    print(f"✗ 训练模式测试失败: {e}")
    exit(1)

print("\n" + "=" * 50)
print("所有测试通过！✓")
print("=" * 50)
print("\n提示：")
print("- 模型已成功初始化并可以正常运行")
print("- 可以使用 example_usage.py 查看更详细的使用示例")
print("- 可以使用 config_example.py 中的配置进行训练")
print("- 首次运行会自动下载CLIP预训练模型（约1GB）")

