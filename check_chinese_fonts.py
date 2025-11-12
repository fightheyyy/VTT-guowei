"""
检查系统可用的中文字体
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

print("\n" + "="*70)
print("检查系统可用的中文字体")
print("="*70 + "\n")

# 获取所有字体
all_fonts = fm.findSystemFonts()
print(f"系统共有 {len(all_fonts)} 个字体\n")

# 查找常见中文字体
chinese_font_names = [
    'Microsoft YaHei',  # 微软雅黑
    'SimHei',           # 黑体
    'SimSun',           # 宋体
    'KaiTi',            # 楷体
    'FangSong',         # 仿宋
    'Arial Unicode MS', # Arial Unicode
    'STHeiti',          # 华文黑体（Mac）
    'PingFang SC',      # 苹方（Mac）
]

print("检查常见中文字体:")
print("-" * 70)

available_fonts = []
for font_name in chinese_font_names:
    try:
        font = fm.FontProperties(family=font_name)
        font_path = fm.findfont(font)
        
        # 检查是否真的找到了该字体（不是回退到默认字体）
        if font_name.lower() in font_path.lower() or 'default' not in font_path.lower():
            print(f"✓ {font_name:20s} - 可用")
            available_fonts.append(font_name)
        else:
            print(f"✗ {font_name:20s} - 不可用")
    except:
        print(f"✗ {font_name:20s} - 不可用")

print("-" * 70)

if available_fonts:
    print(f"\n找到 {len(available_fonts)} 个可用的中文字体:")
    for font in available_fonts:
        print(f"  - {font}")
    
    # 测试绘图
    print(f"\n正在测试中文显示（使用: {available_fonts[0]}）...")
    
    plt.rcParams['font.sans-serif'] = available_fonts
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, '中文显示测试\nChinese Font Test\n这是中文字体', 
            fontsize=20, ha='center', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('中文字体测试', fontsize=16)
    ax.set_xlabel('横轴标签', fontsize=12)
    ax.set_ylabel('纵轴标签', fontsize=12)
    
    test_path = 'chinese_font_test.png'
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 测试图片已保存: {test_path}")
    print(f"\n建议使用的字体配置:")
    print(f"  plt.rcParams['font.sans-serif'] = {available_fonts[:3]}")
    print(f"  plt.rcParams['axes.unicode_minus'] = False")
else:
    print("\n警告：未找到可用的中文字体！")
    print("\n解决方案:")
    print("  1. Windows用户: 系统应该已经有'Microsoft YaHei'或'SimHei'")
    print("  2. 如果字体不可用，请安装中文字体包")
    print("  3. 或者使用英文标签代替中文")

print("\n" + "="*70 + "\n")

