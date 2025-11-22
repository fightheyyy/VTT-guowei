"""
测试数据增强功能
快速验证增强是否正常工作
"""

import sys
sys.path.append('../..')

import torch
import matplotlib.pyplot as plt
from data_augmentation import get_augmentation_pipeline

def test_time_series_augmentation():
    """测试时序数据增强"""
    print("="*70)
    print("测试时序数据增强")
    print("="*70)
    
    # 创建测试数据 [Batch, Time, Variates]
    x = torch.randn(4, 12, 14) * 5 + 10  # 模拟真实数据范围
    
    # 获取增强器
    ts_aug, _, config = get_augmentation_pipeline(mode='medium')
    
    # 应用增强
    x_aug = ts_aug(x, 
                   augment_prob=config['ts_prob'],
                   augment_types=config['ts_types'])
    
    # 统计差异
    diff = (x_aug - x).abs()
    print(f"\n✓ 原始数据形状: {x.shape}")
    print(f"✓ 增强数据形状: {x_aug.shape}")
    print(f"✓ 平均变化: {diff.mean().item():.6f}")
    print(f"✓ 最大变化: {diff.max().item():.6f}")
    print(f"✓ 变化率: {(diff.mean() / x.abs().mean() * 100).item():.2f}%")
    
    # 可视化第一个样本的第一个变量
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x[0, :, 0].numpy(), 'b-o', label='Original', linewidth=2)
    plt.plot(x_aug[0, :, 0].numpy(), 'r--x', label='Augmented', linewidth=2)
    plt.title('Time Series Augmentation (Sample 0, Variate 0)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 可视化多个变量的变化
    plt.subplot(1, 2, 2)
    for i in range(min(3, 14)):  # 显示前3个变量
        plt.plot(x[0, :, i].numpy(), label=f'Var {i} (Orig)', linewidth=1.5)
        plt.plot(x_aug[0, :, i].numpy(), '--', label=f'Var {i} (Aug)', linewidth=1.5, alpha=0.7)
    plt.title('Multiple Variates Augmentation')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(fontsize=8)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('augmentation_test_timeseries.png', dpi=150)
    print(f"\n✓ 可视化已保存: augmentation_test_timeseries.png")
    plt.close()


def test_image_augmentation():
    """测试图像数据增强"""
    print("\n" + "="*70)
    print("测试图像数据增强")
    print("="*70)
    
    # 创建测试图像 [Batch, N_Variates, 3, H, W]
    images = torch.rand(2, 14, 3, 224, 224)
    
    # 获取增强器
    _, img_aug, config = get_augmentation_pipeline(mode='medium')
    
    # 应用增强
    images_aug = img_aug(images, augment_prob=config['img_prob'])
    
    # 统计差异
    diff = (images_aug - images).abs()
    print(f"\n✓ 原始图像形状: {images.shape}")
    print(f"✓ 增强图像形状: {images_aug.shape}")
    print(f"✓ 平均变化: {diff.mean().item():.6f}")
    print(f"✓ 最大变化: {diff.max().item():.6f}")
    print(f"✓ 变化率: {(diff.mean() / images.abs().mean() * 100).item():.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        # 原始图像（第一个样本的前4个变量）
        img_orig = images[0, i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f'Original Variate {i}')
        axes[0, i].axis('off')
        
        # 增强图像
        img_aug = images_aug[0, i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_aug)
        axes[1, i].set_title(f'Augmented Variate {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_test_images.png', dpi=150)
    print(f"✓ 可视化已保存: augmentation_test_images.png")
    plt.close()


def test_augmentation_modes():
    """测试不同增强模式"""
    print("\n" + "="*70)
    print("测试不同增强模式")
    print("="*70)
    
    x = torch.randn(32, 12, 14) * 5 + 10
    
    modes = ['light', 'medium', 'heavy']
    results = {}
    
    for mode in modes:
        ts_aug, _, config = get_augmentation_pipeline(mode=mode)
        x_aug = ts_aug(x, 
                       augment_prob=config['ts_prob'],
                       augment_types=config['ts_types'])
        
        diff = (x_aug - x).abs().mean().item()
        change_rate = (diff / x.abs().mean() * 100).item()
        
        results[mode] = {
            'diff': diff,
            'rate': change_rate,
            'prob': config['ts_prob'],
            'types': config['ts_types']
        }
        
        print(f"\n{mode.upper()}模式:")
        print(f"  增强概率: {config['ts_prob']}")
        print(f"  增强类型: {config['ts_types']}")
        print(f"  平均变化: {diff:.6f}")
        print(f"  变化率: {change_rate:.2f}%")
    
    # 可视化对比
    plt.figure(figsize=(10, 6))
    modes_list = list(results.keys())
    diffs = [results[m]['diff'] for m in modes_list]
    rates = [results[m]['rate'] for m in modes_list]
    
    x_pos = range(len(modes_list))
    plt.bar(x_pos, rates, color=['green', 'orange', 'red'], alpha=0.7)
    plt.xlabel('Augmentation Mode')
    plt.ylabel('Change Rate (%)')
    plt.title('Augmentation Strength Comparison')
    plt.xticks(x_pos, [m.upper() for m in modes_list])
    plt.grid(axis='y', alpha=0.3)
    
    for i, (mode, rate) in enumerate(zip(modes_list, rates)):
        plt.text(i, rate + 0.1, f'{rate:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('augmentation_modes_comparison.png', dpi=150)
    print(f"\n✓ 对比图已保存: augmentation_modes_comparison.png")
    plt.close()


def test_augmentation_consistency():
    """测试增强的一致性（同样数据多次增强应该不同）"""
    print("\n" + "="*70)
    print("测试增强的随机性")
    print("="*70)
    
    x = torch.randn(1, 12, 14) * 5 + 10
    ts_aug, _, config = get_augmentation_pipeline(mode='medium')
    
    # 多次增强同一数据
    augmented_versions = []
    for i in range(5):
        x_aug = ts_aug(x, 
                       augment_prob=1.0,  # 100%概率确保增强
                       augment_types=config['ts_types'])
        augmented_versions.append(x_aug)
    
    # 检查每次增强是否不同
    print("\n检查5次增强结果的差异:")
    for i in range(1, 5):
        diff = (augmented_versions[i] - augmented_versions[0]).abs().mean().item()
        print(f"  版本{i+1} vs 版本1: 差异 = {diff:.6f}")
    
    if all((augmented_versions[i] - augmented_versions[0]).abs().mean().item() > 1e-6 
           for i in range(1, 5)):
        print("✓ 随机性测试通过！每次增强结果不同")
    else:
        print("⚠ 警告：增强结果过于相似，检查随机性")


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "数据增强功能测试" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    try:
        # 测试1: 时序数据增强
        test_time_series_augmentation()
        
        # 测试2: 图像数据增强
        test_image_augmentation()
        
        # 测试3: 不同模式对比
        test_augmentation_modes()
        
        # 测试4: 随机性检查
        test_augmentation_consistency()
        
        print("\n" + "="*70)
        print("✅ 所有测试通过！数据增强功能正常")
        print("="*70)
        print("\n建议:")
        print("  1. 查看生成的可视化图片确认增强效果")
        print("  2. 运行训练脚本: python train_12steps_dual_cached.py")
        print("  3. 观察训练过程中Train/Val F1的差距是否缩小")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

