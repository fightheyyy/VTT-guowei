@echo off
chcp 65001 >nul
echo.
echo ========================================
echo 实验结果查看工具
echo ========================================
echo.
echo 请选择操作:
echo   1. 查看实验摘要
echo   2. 查看Top 5实验
echo   3. 对比数据增强效果
echo   4. 生成完整对比报告
echo   5. 导出论文表格
echo   6. 查看所有实验
echo   0. 退出
echo.
set /p choice=请输入选项 (0-6): 

if "%choice%"=="1" (
    python view_experiments.py --summary
) else if "%choice%"=="2" (
    python view_experiments.py --top 5
) else if "%choice%"=="3" (
    python view_experiments.py --augmentation
) else if "%choice%"=="4" (
    python view_experiments.py --compare
    echo.
    echo ✓ 报告已生成在: experiment_logs/comparison_report.md
    echo ✓ 图表已生成在: experiment_logs/comparison_plots.png
) else if "%choice%"=="5" (
    python view_experiments.py --export
    echo.
    echo ✓ LaTeX表格: experiment_logs/paper_table.tex
    echo ✓ CSV表格: experiment_logs/paper_table.csv
) else if "%choice%"=="6" (
    python view_experiments.py --all
) else if "%choice%"=="0" (
    exit
) else (
    echo 无效选项!
)

echo.
pause

