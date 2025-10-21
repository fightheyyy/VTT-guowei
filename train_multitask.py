"""
多任务训练：一个模型处理多种输入长度
更科学的训练方式
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from models import TimesCLIP
from data_loader_with_yield import create_dataloaders_with_yield


def train_multitask_model(
    csv_path,
    selected_bands=None,
    input_lengths=[9, 12, 15, 18],  # 3,4,5,6个月
    batch_size=8,
    epochs=100,
    d_model=384,
    device='cuda',
    save_dir='checkpoints'
):
    """
    训练一个能处理多种输入长度的通用模型
    
    策略：在训练时随机选择输入长度，让模型学会适应
    """
    print("=" * 70)
    print("多任务训练：训练万能预测模型")
    print("=" * 70)
    print(f"支持的输入长度: {input_lengths}")
    print(f"对应月份数: {[l//3 for l in input_lengths]}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个输入长度创建数据加载器
    dataloaders = {}
    for lookback in input_lengths:
        prediction_steps = 36 - lookback
        train_loader, test_loader, n_variates = create_dataloaders_with_yield(
            csv_path=csv_path,
            selected_bands=selected_bands,
            mode='timeseries',
            lookback=lookback,
            prediction_steps=prediction_steps,
            batch_size=batch_size
        )
        dataloaders[lookback] = {'train': train_loader, 'test': test_loader}
        print(f"  ✓ {lookback}步 → {prediction_steps}步 数据就绪")
    
    # 创建模型（使用最长的配置初始化）
    max_lookback = max(input_lengths)
    model = TimesCLIP(
        time_steps=max_lookback,
        n_variates=n_variates,
        prediction_steps=36 - max_lookback,
        d_model=d_model,
        patch_length=8,
        stride=4
    ).to(device)
    
    # 优化器
    param_groups = model.get_parameter_groups(lr_vision=1e-5, lr_other=5e-5)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    print(f"\n开始多任务训练（共{epochs}个epoch）...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # 每个epoch随机选择一个输入长度训练
        import random
        lookback = random.choice(input_lengths)
        train_loader = dataloaders[lookback]['train']
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (lookback={lookback})')
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # 如果输入长度不是最大值，需要调整模型
            # 这里简化处理：截取或padding
            if x.size(1) < max_lookback:
                # Padding到最大长度
                pad_size = max_lookback - x.size(1)
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_size), value=0)
            
            y_pred, contrastive_loss = model(x, return_loss=True)
            
            # 截取对应长度的预测
            pred_steps = 36 - lookback
            y_pred = y_pred[:, :, :pred_steps]
            
            total_loss_batch, loss_dict = model.compute_loss(
                y_pred, y, contrastive_loss,
                lambda_gen=1.0, lambda_align=0.01
            )
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss_dict['total_loss']
            n_batches += 1
            
            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.2f}"})
        
        # 验证（在所有长度上）
        model.eval()
        val_losses = []
        with torch.no_grad():
            for lookback in input_lengths:
                test_loader = dataloaders[lookback]['test']
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    
                    if x.size(1) < max_lookback:
                        pad_size = max_lookback - x.size(1)
                        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size), value=0)
                    
                    y_pred, contrastive_loss = model(x, return_loss=True)
                    pred_steps = 36 - lookback
                    y_pred = y_pred[:, :, :pred_steps]
                    
                    _, loss_dict = model.compute_loss(
                        y_pred, y, contrastive_loss, 1.0, 0.01
                    )
                    val_losses.append(loss_dict['total_loss'])
        
        avg_train_loss = total_loss / n_batches
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.2f}, Val={avg_val_loss:.2f}")
        
        scheduler.step()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'supported_lengths': input_lengths,
                'config': {
                    'n_variates': n_variates,
                    'd_model': d_model,
                    'max_lookback': max_lookback
                }
            }, os.path.join(save_dir, 'multitask_best.pth'))
            print(f"✓ 保存最佳模型 (Val Loss: {avg_val_loss:.2f})")
    
    print(f"\n多任务训练完成！")
    print(f"该模型可以处理 {[l//3 for l in input_lengths]} 个月的输入")
    return model


if __name__ == "__main__":
    csv_path = "extract2022_20251010_165007.csv"
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = train_multitask_model(
        csv_path=csv_path,
        selected_bands=selected_bands,
        input_lengths=[9, 12, 15, 18],  # 3,4,5,6个月
        batch_size=8,
        epochs=100,
        d_model=384,
        device=device
    )
    
    print("\n✓ 训练完成！现在你有了一个万能模型")
    print("  可以用3、4、5或6个月的数据进行预测！")

