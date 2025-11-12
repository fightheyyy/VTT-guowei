"""
快速测试TimesCLIP实现
验证所有模块能否正常工作
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
print("="*70)
print("TimesCLIP 实现验证")
print("="*70)

# 测试各个模块
batch_size = 2
time_steps = 12
n_variates = 7
d_model = 256

print("\n[1/7] 测试 CLIP-Text 语言模块...")
try:
    from models.language_module_clip import LanguageModuleCLIP
    
    model = LanguageModuleCLIP(
        patch_length=6,
        d_model=d_model,
        freeze_backbone=True
    )
    
    patches = torch.randn(batch_size, n_variates, 3, 6)
    CLS_text, Feat_text = model(patches)
    
    print(f"  ✓ 输入: {patches.shape}")
    print(f"  ✓ CLS输出: {CLS_text.shape}")
    print(f"  ✓ 特征输出: {Feat_text.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ 总参数: {total_params/1e6:.1f}M, 可训练: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n[2/7] 测试对比学习损失...")
try:
    from models.contrastive_loss import InfoNCELoss, HybridContrastiveLoss
    
    CLS_img = torch.randn(batch_size, n_variates, d_model)
    CLS_text = torch.randn(batch_size, n_variates, d_model)
    
    # InfoNCE
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(CLS_img[:, 0, :], CLS_text[:, 0, :])
    print(f"  ✓ InfoNCE损失: {loss.item():.4f}")
    
    # 混合损失
    loss_fn = HybridContrastiveLoss(temperature=0.07, alpha=0.5)
    loss, loss_dict = loss_fn(CLS_img, CLS_text)
    print(f"  ✓ 混合损失: {loss.item():.4f}")
    print(f"    - 变量级: {loss_dict['loss_variate']:.4f}")
    print(f"    - 全局级: {loss_dict['loss_global']:.4f}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n[3/7] 测试变量选择模块...")
try:
    from models.variate_selection_timesclip import VariateSelectionModule, TopKVariateSelection
    
    CLS_img = torch.randn(batch_size, n_variates, d_model)
    CLS_text = torch.randn(batch_size, n_variates, d_model)
    
    # 基础变量选择
    module = VariateSelectionModule(d_model=d_model)
    selected, attn_weights = module(CLS_img, CLS_text, return_weights=True)
    print(f"  ✓ 基础变量选择: {selected.shape}")
    print(f"  ✓ 注意力权重: {attn_weights.shape}")
    
    # Top-K选择
    module = TopKVariateSelection(d_model=d_model, top_k=3)
    selected, indices, scores = module(CLS_img, CLS_text, k=3)
    print(f"  ✓ Top-K选择: {selected.shape}")
    print(f"  ✓ 选中索引: {indices.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n[4/7] 测试预处理器...")
try:
    from models.preprocessor import VisualPreprocessor, LanguagePreprocessor
    
    x = torch.randn(batch_size, time_steps, n_variates)
    
    # 视觉预处理
    visual_prep = VisualPreprocessor(image_size=224)
    images = visual_prep(x)
    print(f"  ✓ 视觉预处理: {x.shape} → {images.shape}")
    
    # 语言预处理
    lang_prep = LanguagePreprocessor(patch_length=6, stride=3)
    patches = lang_prep(x)
    print(f"  ✓ 语言预处理: {x.shape} → {patches.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n[5/7] 测试完整TimesCLIP模型...")
try:
    from models.timesclip_yield_predictor import TimesCLIPYieldPredictor
    
    model = TimesCLIPYieldPredictor(
        time_steps=time_steps,
        n_variates=n_variates,
        d_model=d_model,
        use_variate_selection=True,
        contrastive_weight=0.1
    )
    
    x = torch.randn(batch_size, time_steps, n_variates)
    y = torch.randn(batch_size, 1)
    
    # 前向传播
    y_pred = model(x)
    print(f"  ✓ 前向传播: {x.shape} → {y_pred.shape}")
    
    # 计算损失
    total_loss, loss_dict = model.compute_loss(x, y)
    print(f"  ✓ 总损失: {loss_dict['total_loss']:.4f}")
    print(f"    - 回归: {loss_dict['regression_loss']:.4f}")
    print(f"    - 对比: {loss_dict['contrastive_loss']:.4f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ 总参数: {total_params/1e6:.1f}M")
    print(f"  ✓ 可训练: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n[6/7] 测试纯语言模型...")
try:
    from models.timesclip_yield_predictor import LanguageOnlyTimesCLIPPredictor
    
    model = LanguageOnlyTimesCLIPPredictor(
        time_steps=time_steps,
        n_variates=n_variates,
        d_model=d_model
    )
    
    x = torch.randn(batch_size, time_steps, n_variates)
    y_pred = model(x)
    print(f"  ✓ 前向传播: {x.shape} → {y_pred.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ 总参数: {total_params/1e6:.1f}M, 可训练: {trainable_params/1e6:.1f}M")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n[7/7] 测试训练脚本...")
try:
    import sys
    import subprocess
    
    # 检查数据文件
    data_files = [
        'data/2019产量数据.csv',
        'data/2020产量数据.csv',
        'data/2021产量数据.csv',
        'data/2022产量数据.csv'
    ]
    
    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:
        print(f"  ⚠ 缺少数据文件: {missing_files}")
        print(f"  ℹ 跳过训练测试")
    else:
        print(f"  ✓ 数据文件完整")
        print(f"  ℹ 可以运行: python experiments/yield_prediction/train_timesclip.py --quick")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n" + "="*70)
print("验证完成！")
print("="*70)
print("\n下一步:")
print("1. 快速测试: python experiments/yield_prediction/train_timesclip.py --quick")
print("2. 完整训练: python experiments/yield_prediction/train_timesclip.py --input_steps 12")
print("3. 批处理: run_timesclip.bat")
print("\n详细文档: TIMESCLIP_IMPLEMENTATION.md")

