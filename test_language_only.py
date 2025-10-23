"""
快速测试语言模态版本是否能正常运行
"""

import torch
from models.timesclip_language_only import TimesCLIPLanguageOnly

def test_model():
    print("=" * 60)
    print("测试 TimesCLIPLanguageOnly 模型")
    print("=" * 60)
    
    # 配置
    batch_size = 4
    time_steps = 18
    n_variates = 7
    prediction_steps = 18
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n创建模型...")
    model = TimesCLIPLanguageOnly(
        time_steps=time_steps,
        n_variates=n_variates,
        prediction_steps=prediction_steps,
        patch_length=6,
        stride=3,
        d_model=256,
        n_heads=8
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    x = torch.randn(batch_size, time_steps, n_variates).to(device)
    print(f"输入形状: {x.shape}")
    
    try:
        with torch.no_grad():
            y_pred = model(x)
        print(f"输出形状: {y_pred.shape}")
        print(f"输出范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        print("✓ 前向传播成功")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # 测试反向传播
    print(f"\n测试反向传播...")
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        
        y_true = torch.randn(batch_size, n_variates, prediction_steps).to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"损失值: {loss.item():.6f}")
        print("✓ 反向传播成功")
    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        return False
    
    # 测试特征编码
    print(f"\n测试特征编码...")
    try:
        model.eval()
        with torch.no_grad():
            features = model.encode_timeseries(x)
        print(f"特征形状: {features.shape}")
        print("✓ 特征编码成功")
    except Exception as e:
        print(f"✗ 特征编码失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_model()
    if not success:
        exit(1)

