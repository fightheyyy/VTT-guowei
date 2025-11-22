"""
对比不同训练run的checkpoint
"""
import os
import torch
import glob
import json
from datetime import datetime

def load_checkpoint_info(checkpoint_path):
    """加载checkpoint信息"""
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_f1': checkpoint.get('val_f1', checkpoint.get('best_val_f1', 'N/A')),
        'val_acc': checkpoint.get('val_acc', 'N/A'),
        'path': checkpoint_path
    }

def main():
    # 找到所有12步双模态训练目录
    pattern = "experiments/classification/timesclip_12steps_dual_*"
    dirs = sorted(glob.glob(pattern))
    
    print("="*70)
    print("12步双模态训练历史对比")
    print("="*70)
    print()
    
    if not dirs:
        print("未找到任何训练目录")
        return
    
    results = []
    
    for dir_path in dirs:
        # 提取时间戳
        dir_name = os.path.basename(dir_path)
        timestamp_str = dir_name.replace("timesclip_12steps_dual_", "")
        
        # 尝试加载配置
        config_path = os.path.join(dir_path, "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 加载最佳模型checkpoint
        best_ckpt_path = os.path.join(dir_path, "checkpoints", "best_model.pth")
        best_info = load_checkpoint_info(best_ckpt_path)
        
        # 加载最新checkpoint
        latest_ckpt_path = os.path.join(dir_path, "checkpoints", "latest_checkpoint.pth")
        latest_info = load_checkpoint_info(latest_ckpt_path)
        
        results.append({
            'timestamp': timestamp_str,
            'dir': dir_path,
            'best': best_info,
            'latest': latest_info,
            'config': config
        })
    
    # 显示结果
    print(f"找到 {len(results)} 个训练run:\n")
    
    for i, res in enumerate(results, 1):
        print(f"[{i}] {res['timestamp']}")
        print(f"    目录: {res['dir']}")
        
        if res['best']:
            print(f"    最佳模型: Epoch {res['best']['epoch']}, "
                  f"Val F1={res['best']['val_f1']:.4f}, "
                  f"Acc={res['best']['val_acc']:.4f}")
        else:
            print(f"    最佳模型: 未找到")
        
        if res['latest']:
            print(f"    最新状态: Epoch {res['latest']['epoch']}")
        else:
            print(f"    最新状态: 未找到")
        
        print()
    
    # 找出最佳的
    valid_results = [r for r in results if r['best'] and isinstance(r['best']['val_f1'], float)]
    if valid_results:
        best_run = max(valid_results, key=lambda x: x['best']['val_f1'])
        print("="*70)
        print("🏆 最佳训练run:")
        print(f"   时间: {best_run['timestamp']}")
        print(f"   Val F1: {best_run['best']['val_f1']:.4f}")
        print(f"   Epoch: {best_run['best']['epoch']}")
        print(f"   路径: {best_run['best']['path']}")
        print("="*70)

if __name__ == "__main__":
    main()

