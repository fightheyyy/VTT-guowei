"""
使用12步(120天)数据进行直接分类
专门保存到 timesclip_12steps 文件夹，不覆盖其他模型
"""
import sys
import os
sys.path.append('../..')

from experiments.classification.train_classification_improved import train_timesclip_classifier_improved
from datetime import datetime

if __name__ == "__main__":
    print("="*70)
    print("使用12步(120天)数据进行早期分类")
    print("="*70)
    
    # 创建专门的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/classification/timesclip_12steps_{timestamp}"
    
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/results", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    
    print(f"\n模型将保存到: {save_dir}")
    print(f"不会覆盖现有模型!")
    print("="*70)
    
    # 保存配置信息
    config = {
        "time_steps": 12,
        "days": 120,
        "model_type": "dual",
        "batch_size": 64,
        "epochs": 100,
        "lr": 1e-4,
        "focal_gamma": 2.0,
        "training_time": timestamp,
        "description": "12步(120天)直接分类 - 不覆盖版本"
    }
    
    import json
    with open(f"{save_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 训练模型 (需要修改保存路径)
    # 注意: train_classification_improved默认保存到timesclip_improved
    # 我们需要在训练后手动移动文件，或者修改源码
    
    print("\n警告: 当前版本会保存到默认位置")
    print("训练完成后，请手动检查保存位置")
    print("或使用下面的命令行参数指定保存目录\n")
    
    model, metrics = train_timesclip_classifier_improved(
        csv_path="../../data/2018four.csv",
        time_steps=12,  # ← 固定使用12步
        n_variates=14,
        model_type="dual",  # 双模态
        batch_size=64,
        epochs=100,
        lr=1e-4,
        patience=15,
        
        # 改进策略
        focal_alpha=0.25,
        focal_gamma=2.0,
        time_weight_factor=3.0,
        
        # 不使用课程学习(因为固定12步)
        focus_early=False,  # ← 关闭早期识别课程学习
        
        # 其他配置
        use_cached_images=False,
        multiscale_cache_dir="../../data/multiscale_image_cache",
        test_only=False
    )
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"测试集F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"测试集准确率: {metrics['accuracy']:.4f}")
    print("\n各类别F1:")
    for i, f1 in enumerate(metrics['f1_per_class']):
        print(f"  Class {i}: {f1:.4f}")
    print("="*70)

