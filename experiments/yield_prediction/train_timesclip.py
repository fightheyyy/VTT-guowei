"""
TimesCLIP产量预测训练脚本
完全对齐论文方法
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
from datetime import datetime
import argparse

from models.timesclip_yield_predictor import TimesCLIPYieldPredictor, LanguageOnlyTimesCLIPPredictor
from experiments.yield_prediction.data_loader import create_yield_dataloaders


def train_timesclip_model(
    model, 
    train_loader, 
    test_loader, 
    yield_mean, 
    yield_std,
    epochs=50, 
    lr=1e-4, 
    device='cuda', 
    log_dir='logs', 
    model_name='TimesCLIP',
    use_contrastive=True
):
    """训练TimesCLIP模型"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    writer = SummaryWriter(log_dir)
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 15
    
    print(f"\n开始训练 {model_name}...")
    print(f"- 训练集: {len(train_loader.dataset)}")
    print(f"- 测试集: {len(test_loader.dataset)}")
    print(f"- Epochs: {epochs}")
    print(f"- 对比学习: {'是' if use_contrastive else '否'}")
    print(f"- Device: {device}\n")
    
    for epoch in range(epochs):
        # ===== 训练 =====
        model.train()
        train_loss = 0
        train_reg_loss = 0
        train_contrast_loss = 0
        
        pbar = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/{epochs}', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # TimesCLIP模型支持联合训练
            if use_contrastive and hasattr(model, 'compute_loss'):
                loss, loss_dict = model.compute_loss(x, y)
                train_reg_loss += loss_dict['regression_loss']
                train_contrast_loss += loss_dict['contrastive_loss']
            else:
                y_pred = model(x)
                loss = nn.functional.mse_loss(y_pred, y)
                train_reg_loss += loss.item()
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_reg_loss /= len(train_loader)
        if use_contrastive:
            train_contrast_loss /= len(train_loader)
        
        # ===== 验证 =====
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                if use_contrastive and hasattr(model, 'compute_loss'):
                    loss, loss_dict = model.compute_loss(x, y)
                    val_loss += loss_dict['regression_loss']  # 只用回归损失评估
                    y_pred = model(x, return_contrastive_loss=False)
                else:
                    y_pred = model(x)
                    loss = nn.functional.mse_loss(y_pred, y)
                    val_loss += loss.item()
                
                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        val_loss /= len(test_loader)
        
        # 反归一化计算指标
        all_preds = np.array(all_preds) * yield_std + yield_mean
        all_targets = np.array(all_targets) * yield_std + yield_mean
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        
        # TensorBoard记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_regression', train_reg_loss, epoch)
        if use_contrastive:
            writer.add_scalar('Loss/train_contrastive', train_contrast_loss, epoch)
        writer.add_scalar('Metrics/RMSE', rmse, epoch)
        writer.add_scalar('Metrics/R2', r2, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            if use_contrastive:
                print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Train={train_loss:.4f} (Reg={train_reg_loss:.4f}, Cont={train_contrast_loss:.4f}), Val={val_loss:.4f}, RMSE={rmse:.4f}, R²={r2:.4f} ✓")
            else:
                print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, RMSE={rmse:.4f}, R²={r2:.4f} ✓")
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0:
                print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Val={val_loss:.4f} (patience={patience_counter}/{max_patience})")
        
        if patience_counter >= max_patience:
            print(f"  [{model_name}] 早停于Epoch {epoch+1}, 最佳Val={best_loss:.4f}")
            break
    
    model.load_state_dict(best_model_state)
    writer.close()
    return model


def evaluate_model(model, test_loader, yield_mean, yield_std, device='cuda', use_contrastive=True):
    """评估模型"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            if use_contrastive and hasattr(model, 'forward'):
                y_pred = model(x, return_contrastive_loss=False)
            else:
                y_pred = model(x)
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds) * yield_std + yield_mean
    all_targets = np.array(all_targets) * yield_std + yield_mean
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    mask = all_targets > 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'predictions': all_preds.flatten().tolist(),
        'targets': all_targets.flatten().tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='TimesCLIP产量预测训练')
    parser.add_argument('--quick', action='store_true', help='快速测试（10 epochs）')
    parser.add_argument('--input_steps', type=int, default=12, help='输入时间步数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--no_variate_selection', action='store_true', help='禁用变量选择')
    parser.add_argument('--no_contrastive', action='store_true', help='禁用对比学习')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='对比学习权重')
    parser.add_argument('--language_only', action='store_true', help='只用语言模态（CLIP-Text）')
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 10
        print("\n🚀 快速测试模式")
    
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_steps = args.input_steps
    
    print(f"\n{'='*70}")
    print(f"TimesCLIP 产量预测实验")
    print(f"{'='*70}")
    print(f"输入时间步数: {input_steps} ({input_steps*10}天)")
    print(f"对比学习权重: {args.contrastive_weight}")
    print(f"变量选择: {'否' if args.no_variate_selection else '是'}")
    print(f"对比学习: {'否' if args.no_contrastive else '是'}")
    print(f"模态: {'纯语言（CLIP-Text）' if args.language_only else '双模态（CLIP-Text + CLIP-Vision）'}")
    print(f"设备: {device}")
    print(f"{'='*70}\n")
    
    # 加载数据
    train_loader, test_loader, n_variates, yield_mean, yield_std = create_yield_dataloaders(
        train_csv_paths=[
            'data/2019产量数据.csv',
            'data/2020产量数据.csv',
            'data/2021产量数据.csv'
        ],
        test_csv_paths=['data/2022产量数据.csv'],
        selected_bands=['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red'],
        input_steps=input_steps,
        batch_size=args.batch_size
    )
    
    # 创建模型
    if args.language_only:
        model = LanguageOnlyTimesCLIPPredictor(
            time_steps=input_steps,
            n_variates=n_variates,
            d_model=256,
            patch_length=min(6, input_steps),
            stride=min(3, max(1, input_steps // 2)),
            freeze_clip_text=True
        ).to(device)
        model_name = "TimesCLIP_LanguageOnly"
        use_contrastive = False
    else:
        model = TimesCLIPYieldPredictor(
            time_steps=input_steps,
            n_variates=n_variates,
            d_model=256,
            patch_length=min(6, input_steps),
            stride=min(3, max(1, input_steps // 2)),
            use_variate_selection=not args.no_variate_selection,
            freeze_clip_text=True,
            freeze_clip_vision=True,
            contrastive_weight=args.contrastive_weight if not args.no_contrastive else 0.0
        ).to(device)
        model_name = "TimesCLIP_Full"
        use_contrastive = not args.no_contrastive
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数:")
    print(f"  总参数: {total_params/1e6:.2f}M")
    print(f"  可训练: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    print(f"  冻结: {(total_params-trainable_params)/1e6:.2f}M\n")
    
    # 训练
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'experiments/yield_prediction/timesclip/logs/{model_name}_steps{input_steps}_{timestamp}'
    
    model = train_timesclip_model(
        model, train_loader, test_loader,
        yield_mean, yield_std,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        log_dir=log_dir,
        model_name=model_name,
        use_contrastive=use_contrastive
    )
    
    # 评估
    results = evaluate_model(model, test_loader, yield_mean, yield_std, device, use_contrastive)
    
    # 保存
    os.makedirs('experiments/yield_prediction/timesclip/checkpoints', exist_ok=True)
    os.makedirs('experiments/yield_prediction/timesclip/results', exist_ok=True)
    
    checkpoint_path = f'experiments/yield_prediction/timesclip/checkpoints/{model_name}_steps{input_steps}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_steps': input_steps,
            'n_variates': n_variates,
            'use_variate_selection': not args.no_variate_selection,
            'use_contrastive': use_contrastive,
            'contrastive_weight': args.contrastive_weight
        },
        'results': results,
        'yield_mean': yield_mean,
        'yield_std': yield_std
    }, checkpoint_path)
    
    results_path = f'experiments/yield_prediction/timesclip/results/{model_name}_steps{input_steps}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"训练完成！")
    print(f"{'='*70}")
    print(f"测试集性能:")
    print(f"  - RMSE: {results['rmse']:.4f}")
    print(f"  - MAE:  {results['mae']:.4f}")
    print(f"  - R²:   {results['r2']:.4f}")
    print(f"  - MAPE: {results['mape']:.2f}%")
    print(f"\n保存位置:")
    print(f"  - 模型: {checkpoint_path}")
    print(f"  - 结果: {results_path}")
    print(f"  - 日志: {log_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

