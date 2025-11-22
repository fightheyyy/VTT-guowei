"""
生成汇报用的可视化图表
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_training_history(checkpoint_path):
    """从checkpoint加载训练历史"""
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('history', [])
        return history
    except:
        return None

def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    if not history:
        print("无训练历史数据")
        return
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_f1 = [h['train_f1'] for h in history]
    val_f1 = [h['val_f1'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('训练过程监控', fontsize=16, fontweight='bold')
    
    # 1. Loss曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('损失函数变化', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. F1曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    ax2.plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    
    # 标记最佳点
    best_idx = np.argmax(val_f1)
    best_epoch = epochs[best_idx]
    best_f1 = val_f1[best_idx]
    ax2.plot(best_epoch, best_f1, 'r*', markersize=20, label=f'Best (Epoch {best_epoch})')
    ax2.annotate(f'F1={best_f1:.4f}', 
                xy=(best_epoch, best_f1),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1分数变化', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    ax3.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('准确率变化', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. 过拟合检测
    ax4 = axes[1, 1]
    overfit_gap = np.array(train_f1) - np.array(val_f1)
    ax4.plot(epochs, overfit_gap, 'g-', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.fill_between(epochs, 0, overfit_gap, where=(overfit_gap>0), 
                     color='red', alpha=0.3, label='过拟合区域')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Train F1 - Val F1', fontsize=12)
    ax4.set_title('过拟合监控', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()

def plot_class_distribution(save_dir):
    """绘制类别分布"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ['类别0', '类别1', '类别2', '类别3']
    train_counts = [483, 871, 927, 1719]
    val_counts = [54, 97, 103, 191]
    test_counts = [134, 242, 258, 478]
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, train_counts, width, label='训练集', color='skyblue')
    ax.bar(x, val_counts, width, label='验证集', color='lightgreen')
    ax.bar(x + width, test_counts, width, label='测试集', color='salmon')
    
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('样本数', fontsize=12)
    ax.set_title('数据集类别分布', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (t, v, te) in enumerate(zip(train_counts, val_counts, test_counts)):
        ax.text(i - width, t + 20, str(t), ha='center', fontsize=9)
        ax.text(i, v + 5, str(v), ha='center', fontsize=9)
        ax.text(i + width, te + 10, str(te), ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 类别分布已保存: {save_path}")
    plt.close()

def plot_model_architecture(save_dir):
    """绘制模型架构示意图（简化版）"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 使用文本绘制架构
    architecture_text = """
TimesCLIPClassifier 双模态架构

输入: [Batch, 12步, 14变量]
    │
    ├─────────────────┬──────────────────┐
    │                 │                  │
视觉分支         语言分支         变量选择
CLIP-ViT-B/16   CLIP-Text        自适应加权
(预训练冻结)    + Transformer     (可训练)
    │                 │                  │
    └─────────────────┴──────────────────┘
                      │
              融合特征 (1536维)
                      │
                  MLP分类头
                      │
               4类别输出

参数统计:
- 总参数: 129.15M
- 可训练: 5.52M (4.3%)
- 预训练冻结: 123.63M

创新点:
✓ 双模态表征 (视觉+语言)
✓ 对比学习 (InfoNCE)
✓ 预缓存图像 (77,798张)
"""
    
    ax.text(0.5, 0.5, architecture_text, 
            ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_architecture.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 模型架构已保存: {save_path}")
    plt.close()

def main():
    # 找到最新的训练目录
    import glob
    pattern = "experiments/classification/timesclip_12steps_dual_*"
    dirs = sorted(glob.glob(pattern))
    
    if not dirs:
        print("❌ 未找到训练目录")
        return
    
    latest_dir = dirs[-1]
    print(f"\n使用训练目录: {latest_dir}\n")
    
    # 创建输出目录
    output_dir = os.path.join(latest_dir, "report_figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练历史
    checkpoint_path = os.path.join(latest_dir, "checkpoints", "latest_checkpoint.pth")
    history = load_training_history(checkpoint_path)
    
    print("正在生成汇报图表...")
    print("="*50)
    
    # 生成图表
    if history:
        plot_training_curves(history, output_dir)
    else:
        print("⚠ 未找到训练历史，跳过训练曲线图")
    
    plot_class_distribution(output_dir)
    plot_model_architecture(output_dir)
    
    print("="*50)
    print(f"\n✅ 所有图表已保存至: {output_dir}")
    print("\n生成的文件:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
    print("\n可用于PPT或论文汇报！")

if __name__ == "__main__":
    main()

