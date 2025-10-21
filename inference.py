"""
TimesCLIP模型推理脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models import TimesCLIP
from data_loader import create_dataloaders


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 初始化模型
    model = TimesCLIP(
        time_steps=config['lookback'],
        n_variates=config['n_variates'],
        prediction_steps=config['prediction_steps'],
        patch_length=config['patch_length'],
        stride=config['stride'],
        d_model=config['d_model']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功！")
    print(f"训练epoch: {checkpoint['epoch']}")
    print(f"验证损失: {checkpoint['val_loss']:.4f}")
    
    return model, config


def predict(model, x, device='cuda'):
    """
    单个样本预测
    
    参数:
        model: TimesCLIP模型
        x: 输入数据 [lookback, n_variates]
        device: 设备
    
    返回:
        预测结果 [n_variates, prediction_steps]
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(x).unsqueeze(0).to(device)  # [1, lookback, n_variates]
        y_pred = model(x, return_loss=False)  # [1, n_variates, prediction_steps]
        y_pred = y_pred.squeeze(0).cpu().numpy()  # [n_variates, prediction_steps]
    
    return y_pred


def visualize_prediction(x, y_true, y_pred, band_names, save_path=None):
    """
    可视化预测结果
    
    参数:
        x: 输入序列 [lookback, n_variates]
        y_true: 真实值 [n_variates, prediction_steps]
        y_pred: 预测值 [n_variates, prediction_steps]
        band_names: 波段名称列表
        save_path: 保存路径
    """
    n_variates = x.shape[1]
    lookback = x.shape[0]
    prediction_steps = y_pred.shape[1]
    
    fig, axes = plt.subplots(n_variates, 1, figsize=(15, 3*n_variates))
    if n_variates == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # 输入序列
        ax.plot(range(lookback), x[:, i], 'b-', label='输入序列', linewidth=2)
        
        # 真实值
        ax.plot(range(lookback, lookback + prediction_steps), 
                y_true[i, :], 'g-', label='真实值', linewidth=2)
        
        # 预测值
        ax.plot(range(lookback, lookback + prediction_steps), 
                y_pred[i, :], 'r--', label='预测值', linewidth=2)
        
        # 垂直线分隔输入和预测
        ax.axvline(x=lookback, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(f'{band_names[i]} 波段', fontsize=12)
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('值', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(model, test_loader, device='cuda', n_samples=5, band_names=None):
    """
    评估模型并可视化部分结果
    """
    model.eval()
    
    all_mse = []
    all_mae = []
    
    print("正在评估模型...")
    
    with torch.no_grad():
        for batch_idx, (x, y_true) in enumerate(test_loader):
            x = x.to(device)
            y_true_tensor = y_true.to(device)
            
            # 预测
            y_pred = model(x, return_loss=False)
            
            # 计算指标
            mse = torch.mean((y_pred - y_true_tensor) ** 2).item()
            mae = torch.mean(torch.abs(y_pred - y_true_tensor)).item()
            
            all_mse.append(mse)
            all_mae.append(mae)
            
            # 可视化前几个样本
            if batch_idx < n_samples:
                x_np = x[0].cpu().numpy()
                y_true_np = y_true[0].cpu().numpy()
                y_pred_np = y_pred[0].cpu().numpy()
                
                visualize_prediction(
                    x_np, y_true_np, y_pred_np, band_names,
                    save_path=f'predictions/sample_{batch_idx+1}.png'
                )
    
    # 统计结果
    mean_mse = np.mean(all_mse)
    mean_mae = np.mean(all_mae)
    
    print(f"\n评估结果:")
    print(f"平均MSE: {mean_mse:.4f}")
    print(f"平均MAE: {mean_mae:.4f}")
    print(f"RMSE: {np.sqrt(mean_mse):.4f}")
    
    return mean_mse, mean_mae


if __name__ == "__main__":
    import os
    
    # 创建预测结果目录
    os.makedirs('predictions', exist_ok=True)
    
    # 配置
    csv_path = "extract2022_20251010_165007.csv"
    checkpoint_path = "checkpoints/best_model.pth"
    selected_bands = ['NIR', 'RVI', 'SWIR1', 'blue', 'evi', 'ndvi', 'red']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    model, config = load_model(checkpoint_path, device)
    
    # 创建数据加载器
    _, test_loader, _ = create_dataloaders(
        csv_path=csv_path,
        selected_bands=selected_bands,
        lookback=config['lookback'],
        prediction_steps=config['prediction_steps'],
        batch_size=16
    )
    
    # 评估模型
    evaluate_model(
        model, test_loader, device, 
        n_samples=5, 
        band_names=selected_bands
    )
    
    print("\n推理完成！")

