"""
对比双模态版本和语言模态版本的训练结果
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import TimesCLIP
from models.timesclip_language_only import TimesCLIPLanguageOnly
from models.yield_predictor import YieldPredictor
from data_loader_multiyear import create_multiyear_dataloaders
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def evaluate_stage1(model, test_loader, device='cuda'):
    """评估阶段1模型"""
    model.eval()
    mse_loss = torch.nn.MSELoss()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y_true in test_loader:
            x = x.to(device)
            y_true = y_true.to(device)
            
            # 根据模型类型调用
            if isinstance(model, TimesCLIP):
                y_pred, _ = model(x)  # 双模态版本返回预测和对齐损失
            else:
                y_pred = model(x)  # 语言模态版本只返回预测
            
            loss = mse_loss(y_pred, y_true)
            total_loss += loss.item()
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    rmse = np.sqrt(avg_loss)
    
    # 计算MAE
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return {
        'mse': avg_loss,
        'rmse': rmse,
        'mae': mae
    }


def evaluate_stage2(model, test_loader, device='cuda'):
    """评估阶段2模型"""
    model.eval()
    mse_loss = torch.nn.MSELoss()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y_true in test_loader:
            x = x.to(device)
            y_true = y_true.to(device)
            
            y_pred = model(x)
            
            loss = mse_loss(y_pred, y_true)
            total_loss += loss.item()
            
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y_true.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    # 计算R²
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # MAE
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return {
        'mse': avg_loss,
        'rmse': np.sqrt(avg_loss),
        'r2': r2,
        'mae': mae
    }


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 70)
    print("对比双模态版本 vs 语言模态版本")
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
    
    # 加载测试数据
    print("\n加载测试数据...")
    _, test_loader_stage1, n_variates = create_multiyear_dataloaders(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        mode='timeseries',
        lookback=18,
        prediction_steps=18,
        batch_size=8
    )
    
    _, test_loader_stage2, _ = create_multiyear_dataloaders(
        train_csv_paths=train_files,
        test_csv_paths=test_files,
        selected_bands=selected_bands,
        mode='yield',
        batch_size=8
    )
    
    results = {}
    
    # ========== 评估阶段1 ==========
    print("\n" + "=" * 70)
    print("阶段1：时间序列补全")
    print("=" * 70)
    
    # 双模态版本
    print("\n[1/2] 评估双模态版本...")
    if os.path.exists('checkpoints/stage1_timeseries_best.pth'):
        model_both = TimesCLIP(
            time_steps=18,
            n_variates=n_variates,
            prediction_steps=18,
            patch_length=6,
            stride=3,
            d_model=256
        ).to(device)
        model_both.load_state_dict(torch.load('checkpoints/stage1_timeseries_best.pth'))
        
        results['stage1_both'] = evaluate_stage1(model_both, test_loader_stage1, device)
        total_params, trainable_params = count_parameters(model_both)
        results['stage1_both']['total_params'] = total_params
        results['stage1_both']['trainable_params'] = trainable_params
        
        print(f"  MSE Loss: {results['stage1_both']['mse']:.6f}")
        print(f"  RMSE: {results['stage1_both']['rmse']:.4f}")
        print(f"  MAE: {results['stage1_both']['mae']:.4f}")
        print(f"  参数量: {total_params:,}")
    else:
        print("  未找到模型文件: checkpoints/stage1_timeseries_best.pth")
        results['stage1_both'] = None
    
    # 语言模态版本
    print("\n[2/2] 评估语言模态版本...")
    if os.path.exists('checkpoints/stage1_language_only_best.pth'):
        model_language = TimesCLIPLanguageOnly(
            time_steps=18,
            n_variates=n_variates,
            prediction_steps=18,
            patch_length=6,
            stride=3,
            d_model=256
        ).to(device)
        model_language.load_state_dict(torch.load('checkpoints/stage1_language_only_best.pth'))
        
        results['stage1_language'] = evaluate_stage1(model_language, test_loader_stage1, device)
        total_params, trainable_params = count_parameters(model_language)
        results['stage1_language']['total_params'] = total_params
        results['stage1_language']['trainable_params'] = trainable_params
        
        print(f"  MSE Loss: {results['stage1_language']['mse']:.6f}")
        print(f"  RMSE: {results['stage1_language']['rmse']:.4f}")
        print(f"  MAE: {results['stage1_language']['mae']:.4f}")
        print(f"  参数量: {total_params:,}")
    else:
        print("  未找到模型文件: checkpoints/stage1_language_only_best.pth")
        results['stage1_language'] = None
    
    # ========== 评估阶段2 ==========
    print("\n" + "=" * 70)
    print("阶段2：产量预测")
    print("=" * 70)
    
    # 双模态版本
    print("\n[1/2] 评估双模态版本...")
    if os.path.exists('checkpoints/stage2_yield_best.pth'):
        model_yield_both = YieldPredictor(
            n_variates=n_variates,
            time_steps=36,
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=0.3
        ).to(device)
        model_yield_both.load_state_dict(torch.load('checkpoints/stage2_yield_best.pth'))
        
        results['stage2_both'] = evaluate_stage2(model_yield_both, test_loader_stage2, device)
        total_params, trainable_params = count_parameters(model_yield_both)
        results['stage2_both']['total_params'] = total_params
        
        print(f"  MSE Loss: {results['stage2_both']['mse']:.6f}")
        print(f"  RMSE: {results['stage2_both']['rmse']:.4f}")
        print(f"  MAE: {results['stage2_both']['mae']:.4f}")
        print(f"  R² Score: {results['stage2_both']['r2']:.4f}")
        print(f"  参数量: {total_params:,}")
    else:
        print("  未找到模型文件: checkpoints/stage2_yield_best.pth")
        results['stage2_both'] = None
    
    # 语言模态版本
    print("\n[2/2] 评估语言模态版本...")
    if os.path.exists('checkpoints/stage2_language_only_best.pth'):
        model_yield_language = YieldPredictor(
            n_variates=n_variates,
            time_steps=36,
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=0.3
        ).to(device)
        model_yield_language.load_state_dict(torch.load('checkpoints/stage2_language_only_best.pth'))
        
        results['stage2_language'] = evaluate_stage2(model_yield_language, test_loader_stage2, device)
        total_params, trainable_params = count_parameters(model_yield_language)
        results['stage2_language']['total_params'] = total_params
        
        print(f"  MSE Loss: {results['stage2_language']['mse']:.6f}")
        print(f"  RMSE: {results['stage2_language']['rmse']:.4f}")
        print(f"  MAE: {results['stage2_language']['mae']:.4f}")
        print(f"  R² Score: {results['stage2_language']['r2']:.4f}")
        print(f"  参数量: {total_params:,}")
    else:
        print("  未找到模型文件: checkpoints/stage2_language_only_best.pth")
        results['stage2_language'] = None
    
    # ========== 对比总结 ==========
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    
    if results['stage1_both'] and results['stage1_language']:
        print("\n阶段1（时间序列补全）:")
        print(f"  指标              双模态          语言模态        差异")
        print(f"  {'─'*60}")
        
        mse_diff = (results['stage1_language']['mse'] - results['stage1_both']['mse']) / results['stage1_both']['mse'] * 100
        rmse_diff = (results['stage1_language']['rmse'] - results['stage1_both']['rmse']) / results['stage1_both']['rmse'] * 100
        mae_diff = (results['stage1_language']['mae'] - results['stage1_both']['mae']) / results['stage1_both']['mae'] * 100
        params_diff = (results['stage1_language']['total_params'] - results['stage1_both']['total_params']) / results['stage1_both']['total_params'] * 100
        
        print(f"  MSE Loss:        {results['stage1_both']['mse']:.6f}     {results['stage1_language']['mse']:.6f}     {mse_diff:+.1f}%")
        print(f"  RMSE:            {results['stage1_both']['rmse']:.4f}       {results['stage1_language']['rmse']:.4f}       {rmse_diff:+.1f}%")
        print(f"  MAE:             {results['stage1_both']['mae']:.4f}       {results['stage1_language']['mae']:.4f}       {mae_diff:+.1f}%")
        print(f"  参数量:          {results['stage1_both']['total_params']:,}  {results['stage1_language']['total_params']:,}  {params_diff:+.1f}%")
        
        print(f"\n  性能保留率: {100 - abs(rmse_diff):.1f}% (RMSE)")
        
        if abs(rmse_diff) < 5:
            print(f"  结论: ⭐ 语言模态性能相当（差异<5%），建议去掉视觉模态")
        elif abs(rmse_diff) < 10:
            print(f"  结论: ⚠ 语言模态性能略有差异（5-10%），需权衡")
        else:
            print(f"  结论: ❌ 视觉模态显著提升性能（>10%），建议保留")
    
    if results['stage2_both'] and results['stage2_language']:
        print("\n阶段2（产量预测）:")
        print(f"  指标              双模态          语言模态        差异")
        print(f"  {'─'*60}")
        
        mse_diff = (results['stage2_language']['mse'] - results['stage2_both']['mse']) / results['stage2_both']['mse'] * 100
        r2_diff = (results['stage2_language']['r2'] - results['stage2_both']['r2']) / results['stage2_both']['r2'] * 100
        
        print(f"  MSE Loss:        {results['stage2_both']['mse']:.6f}     {results['stage2_language']['mse']:.6f}     {mse_diff:+.1f}%")
        print(f"  RMSE:            {results['stage2_both']['rmse']:.4f}       {results['stage2_language']['rmse']:.4f}")
        print(f"  R² Score:        {results['stage2_both']['r2']:.4f}       {results['stage2_language']['r2']:.4f}       {r2_diff:+.1f}%")
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    
    # 保存结果
    import json
    with open('comparison_results.json', 'w', encoding='utf-8') as f:
        # 将numpy类型转换为python类型
        results_serializable = {}
        for k, v in results.items():
            if v is not None:
                results_serializable[k] = {
                    key: float(val) if isinstance(val, (np.floating, np.integer)) else val
                    for key, val in v.items()
                }
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print("\n结果已保存到: comparison_results.json")


if __name__ == "__main__":
    main()

