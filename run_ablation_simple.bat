@echo off
chcp 65001 >nul
REM 设置UTF-8编码，避免中文乱码

echo.
echo ========================================================================
echo 消融实验：自动化训练和对比
echo ========================================================================
echo.

REM 创建日志目录
if not exist "experiment_logs" mkdir experiment_logs

echo [1/3] 训练双模态版本 (预计 8-10 小时)...
echo 开始时间: %date% %time%
echo.

python train_multiyear_mirror.py

if %errorlevel% neq 0 (
    echo.
    echo 错误: 双模态训练失败！
    pause
    exit /b 1
)

echo.
echo 双模态训练完成！
echo 完成时间: %date% %time%
echo.

echo [2/3] 训练语言模态版本 (预计 2-3 小时)...
echo 开始时间: %date% %time%
echo.

python train_language_only.py

if %errorlevel% neq 0 (
    echo.
    echo 错误: 语言模态训练失败！
    pause
    exit /b 1
)

echo.
echo 语言模态训练完成！
echo 完成时间: %date% %time%
echo.

echo [3/3] 对比结果...
echo.

python compare_results.py

echo.
echo ========================================================================
echo 实验全部完成！
echo ========================================================================
echo.
echo 结果文件:
echo   - checkpoints\stage1_timeseries_best.pth
echo   - checkpoints\stage1_language_only_best.pth
echo   - comparison_results.json
echo.

pause

