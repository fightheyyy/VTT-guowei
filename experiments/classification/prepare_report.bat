@echo off
chcp 65001 >nul
echo ====================================
echo 汇报材料准备工具
echo ====================================
echo.

echo [1/3] 检查训练目录...
if not exist "experiments\classification\timesclip_12steps_dual_*" (
    echo ❌ 未找到训练目录
    echo 请先运行训练脚本
    pause
    exit
)

for /f "delims=" %%i in ('dir /b /ad /o-d "experiments\classification\timesclip_12steps_dual_*" 2^>nul') do (
    set latest_dir=%%i
    goto :found
)

:found
echo ✓ 找到训练目录: %latest_dir%
echo.

echo [2/3] 生成可视化图表...
cd /d "%~dp0"
python generate_report_figures.py
if errorlevel 1 (
    echo ⚠ 图表生成失败，继续...
) else (
    echo ✓ 图表生成完成
)
echo.

echo [3/3] 准备汇报材料清单...
echo.
echo ====================================
echo 汇报材料已准备完成！
echo ====================================
echo.
echo 📁 文档类:
echo   ✓ 训练策略汇报.md          [完整版]
echo   ✓ 汇报要点-简版.md          [PPT版]
echo   ✓ 汇报材料使用指南.md       [使用说明]
echo.
echo 📊 可视化类:
echo   ✓ training_curves.png       [训练曲线]
echo   ✓ class_distribution.png    [类别分布]
echo   ✓ model_architecture.png    [模型架构]
echo.
echo 💾 数据类:
echo   ✓ best_model.pth            [最佳模型]
echo   ✓ config.json               [训练配置]
echo.
echo 📂 所有材料位于:
echo   - 文档: experiments/classification/
echo   - 图表: experiments/classification/%latest_dir%/report_figures/
echo   - 模型: experiments/classification/%latest_dir%/checkpoints/
echo.
echo ====================================
echo 下一步操作:
echo ====================================
echo.
echo [选项1] 口头汇报 (15分钟)
echo   → 打开: 汇报要点-简版.md
echo   → 准备: training_curves.png + class_distribution.png
echo.
echo [选项2] 书面报告
echo   → 使用: 训练策略汇报.md
echo   → 插入: 所有生成的图表
echo.
echo [选项3] PPT制作
echo   → 参考: 汇报材料使用指南.md (PPT制作流程)
echo.
echo ====================================
echo.
set /p open_guide=是否打开使用指南? [Y/N]: 
if /i "%open_guide%"=="y" (
    start 汇报材料使用指南.md
)
echo.
pause

