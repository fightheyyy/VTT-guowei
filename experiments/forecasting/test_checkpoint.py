"""
测试已保存的checkpoint
快速评估模型效果，无需重新训练
"""

import sys
sys.path.append('../..')

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from experiments.forecasting.two_stage_pipeline import TwoStageDataset, TwoStagePipeline, evaluate_pipeline
from models.timesclip_forecaster import TimesCLIPForecaster
from models.timesclip_classifier import TimesCLIPClassifier


def load_data(csv_path, time_steps=37, n_variates=14):
    """加载数据"""
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    n_samples = X.shape[0]
    X = X.reshape(n_samples, time_steps, n_variates)
    
    # 标准化
    X_normalized = np.zeros_like(X)
    for i in range(n_variates):
        variate_data = X[:, :, i]
        mean = variate_data.mean()
        std = variate_data.std() + 1e-8
        X_normalized[:, :, i] = (variate_data - mean) / std
    
    return X_normalized, y


def test_checkpoint(checkpoint_path='experiments/forecasting/checkpoints/pipeline_e2e_in6.pth'):
    """
    测试已保存的checkpoint
    """
    
    print("="*70)
    print("测试已保存的Checkpoint")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 检查checkpoint是否存在
    import os
    if not os.path.exists(checkpoint_path):
        print(f"\n错误: 未找到checkpoint文件")
        print(f"路径: {checkpoint_path}")
        return
    
    # 加载checkpoint
    print(f"\n加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 显示checkpoint信息
    if isinstance(checkpoint, dict):
        print("\nCheckpoint信息:")
        if 'epoch' in checkpoint:
            print(f"  训练轮次: Epoch {checkpoint['epoch']}")
        if 'best_val_f1' in checkpoint:
            print(f"  最佳验证F1: {checkpoint['best_val_f1']:.4f}")
        if 'config' in checkpoint:
            print(f"  配置: {checkpoint['config']}")
        
        # 提取配置
        config = checkpoint.get('config', {})
        input_len = config.get('input_len', 6)
        output_len = config.get('output_len', 37)
        decoder_type = config.get('decoder_type', 'transformer')
    else:
        print("\n旧格式checkpoint（仅包含模型权重）")
        input_len = 6
        output_len = 37
        decoder_type = 'transformer'
    
    # 加载数据
    print(f"\n加载数据...")
    X, y = load_data('../../data/2018four.csv', output_len, 14)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"测试集样本数: {len(X_test)}")
    
    # 创建数据加载器
    test_dataset = TwoStageDataset(X_test, y_test, input_len, output_len)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建模型
    print(f"\n创建模型...")
    print(f"  输入长度: {input_len}步")
    print(f"  解码器类型: {decoder_type}")
    
    forecaster = TimesCLIPForecaster(
        input_len=input_len,
        output_len=output_len,
        n_variates=14,
        decoder_type=decoder_type,
        use_vision=False,
        use_language=True,
        patch_length=2,
        stride=1
    ).to(device)
    
    classifier = TimesCLIPClassifier(
        time_steps=output_len,
        n_variates=14,
        num_classes=4,
        patch_length=4,
        stride=4
    ).to(device)
    
    pipeline = TwoStagePipeline(forecaster, classifier).to(device)
    
    # 加载权重
    print(f"\n加载模型权重...")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pipeline.load_state_dict(checkpoint['model_state_dict'])
    else:
        pipeline.load_state_dict(checkpoint)
    
    # 评估
    print(f"\n开始评估...")
    test_metrics = evaluate_pipeline(pipeline, test_loader, device)
    
    # 显示结果
    print("\n" + "="*70)
    print("测试集结果")
    print("="*70)
    print(f"F1 Score (macro): {test_metrics['f1_macro']:.4f}")
    print(f"准确率:          {test_metrics['accuracy']:.4f}")
    print("="*70)
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(
        test_metrics['labels'], 
        test_metrics['predictions'],
        target_names=[f'类别{i}' for i in range(4)]
    ))
    
    # 混淆矩阵
    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    print("\n混淆矩阵:")
    print(cm)
    
    # 对比baseline
    baseline_f1 = 0.10  # 直接分类的F1
    improvement = (test_metrics['f1_macro'] / baseline_f1) - 1
    print("\n" + "="*70)
    print("vs Baseline对比:")
    print("="*70)
    print(f"Baseline F1 (直接分类):  0.1000")
    print(f"当前模型F1:             {test_metrics['f1_macro']:.4f}")
    print(f"提升倍数:               {test_metrics['f1_macro']/baseline_f1:.2f}×")
    print(f"相对提升:               {improvement*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试已保存的checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       default='experiments/forecasting/checkpoints/pipeline_e2e_in6.pth',
                       help='checkpoint路径')
    
    args = parser.parse_args()
    
    test_checkpoint(args.checkpoint)

