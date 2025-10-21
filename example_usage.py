"""
TimesCLIP模型使用示例
演示如何初始化模型、进行前向传播和训练
"""

import torch
import torch.nn as nn
from models import TimesCLIP


def example_forward():
    """演示基本的前向传播"""
    print("=== 示例1: 基本前向传播 ===")
    
    # 初始化模型
    model = TimesCLIP(
        time_steps=96,
        n_variates=7,
        prediction_steps=96,
        patch_length=16,
        stride=8,
        d_model=512,
        n_heads=8
    )
    
    # 创建模拟输入数据
    batch_size = 4
    x = torch.randn(batch_size, 96, 7)  # [Batch, Time_Steps, N_Variates]
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        y_pred, contrastive_loss = model(x, return_loss=True)
    
    print(f"预测输出形状: {y_pred.shape}")
    print(f"对比损失: {contrastive_loss.item():.4f}")
    print()


def example_training_step():
    """演示一个训练步骤"""
    print("=== 示例2: 训练步骤 ===")
    
    # 初始化模型
    model = TimesCLIP(
        time_steps=96,
        n_variates=7,
        prediction_steps=96,
        d_model=512
    )
    
    # 获取差异化学习率的参数组
    param_groups = model.get_parameter_groups(
        lr_vision=1e-5,
        lr_other=1e-4
    )
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(param_groups)
    
    # 模拟训练数据
    batch_size = 8
    x = torch.randn(batch_size, 96, 7)  # 输入
    y_true = torch.randn(batch_size, 7, 96)  # 真实标签
    
    # 训练模式
    model.train()
    
    # 前向传播
    y_pred, contrastive_loss = model(x, return_loss=True)
    
    # 计算总损失
    total_loss, loss_dict = model.compute_loss(
        y_pred=y_pred,
        y_true=y_true,
        contrastive_loss=contrastive_loss,
        lambda_gen=1.0,
        lambda_align=0.1
    )
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"总损失: {loss_dict['total_loss']:.4f}")
    print(f"MSE损失: {loss_dict['mse_loss']:.4f}")
    print(f"对比损失: {loss_dict['contrastive_loss']:.4f}")
    print()


def example_inference():
    """演示推理过程"""
    print("=== 示例3: 推理 ===")
    
    model = TimesCLIP(
        time_steps=96,
        n_variates=7,
        prediction_steps=96
    )
    
    # 推理模式
    model.eval()
    
    # 单个样本
    x = torch.randn(1, 96, 7)
    
    with torch.no_grad():
        y_pred = model(x, return_loss=False)
    
    print(f"输入形状: {x.shape}")
    print(f"预测形状: {y_pred.shape}")
    print(f"预测值范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print()


def example_model_info():
    """显示模型信息"""
    print("=== 示例4: 模型信息 ===")
    
    model = TimesCLIP(
        time_steps=96,
        n_variates=7,
        prediction_steps=96
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {frozen_params:,}")
    print()
    
    # 显示各模块参数量
    print("各模块参数量:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module_params:,}")


if __name__ == "__main__":
    print("\nTimesCLIP 模型使用示例\n")
    print("注意: 首次运行会下载CLIP预训练模型（约1GB）\n")
    
    # 运行各个示例
    example_forward()
    example_training_step()
    example_inference()
    example_model_info()
    
    print("所有示例运行完成！")

