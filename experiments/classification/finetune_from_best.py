"""
从最佳模型开始微调
适用于：最佳模型出现较早，后续epoch过拟合的情况
"""
import argparse
import torch
import glob
import os
from train_12steps_dual_cached import main as train_main

def find_best_checkpoint():
    """找到所有训练中最好的checkpoint"""
    pattern = "experiments/classification/timesclip_12steps_dual_*/checkpoints/best_model.pth"
    checkpoints = glob.glob(pattern)
    
    best_f1 = 0
    best_path = None
    
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            f1 = ckpt.get('val_f1', 0)
            epoch = ckpt.get('epoch', 0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_path = ckpt_path
                best_epoch = epoch
        except:
            continue
    
    return best_path, best_f1, best_epoch

def main():
    parser = argparse.ArgumentParser(description='从最佳模型开始微调')
    parser.add_argument('--best_checkpoint', type=str, default=None,
                        help='最佳模型路径（不指定则自动寻找）')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='微调学习率（建议比初始lr小10倍）')
    args = parser.parse_args()
    
    # 找到最佳checkpoint
    if args.best_checkpoint is None:
        best_path, best_f1, best_epoch = find_best_checkpoint()
        if best_path is None:
            print("❌ 未找到任何最佳模型checkpoint")
            return
        print(f"\n✓ 找到最佳模型:")
        print(f"  路径: {best_path}")
        print(f"  Val F1: {best_f1:.4f}")
        print(f"  Epoch: {best_epoch}")
    else:
        best_path = args.best_checkpoint
    
    print(f"\n{'='*70}")
    print("从最佳模型开始微调")
    print(f"{'='*70}")
    print(f"最佳模型: {best_path}")
    print(f"微调学习率: {args.lr}")
    print()
    print("⚠ 注意: 这会创建一个新的训练目录")
    print("原始训练不受影响")
    print(f"{'='*70}\n")
    
    confirm = input("确认开始微调? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("取消")
        return
    
    # TODO: 实现微调逻辑
    # 1. 加载best_model.pth的模型权重
    # 2. 创建新的优化器（使用更小的学习率）
    # 3. 重新初始化学习率调度器
    # 4. 开始新的训练run
    
    print("\n[功能开发中]")
    print("建议: 手动复制train_12steps_dual_cached.py")
    print("     修改LR为1e-5，并在main()开始处添加：")
    print("     checkpoint = torch.load('path/to/best_model.pth')")
    print("     model.load_state_dict(checkpoint['model_state_dict'])")

if __name__ == "__main__":
    main()

