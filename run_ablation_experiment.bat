@echo off
REM ========================================================================
REM 自动化消融实验脚本
REM 功能：依次训练双模态、语言模态，然后对比结果
REM ========================================================================

echo.
echo ========================================================================
echo 消融实验：对比双模态 vs 语言模态
echo ========================================================================
echo.

REM 创建日志目录
if not exist "experiment_logs" mkdir experiment_logs

REM 获取时间戳
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/: " %%a in ('time /t') do (set mytime=%%a%%b)
set timestamp=%mydate%_%mytime%

echo 实验开始时间: %date% %time%
echo 日志保存在: experiment_logs\ablation_experiment_%timestamp%.log
echo.

REM ========================================================================
REM 步骤1: 训练双模态版本
REM ========================================================================
echo.
echo [1/3] 训练双模态版本 (预计 8-10 小时)...
echo ========================================================================
echo 开始时间: %date% %time%
echo.

python train_multiyear_mirror.py > experiment_logs\step1_both_modal_%timestamp%.log 2>&1

if %errorlevel% neq 0 (
    echo.
    echo ❌ 错误: 双模态训练失败！
    echo 请查看日志: experiment_logs\step1_both_modal_%timestamp%.log
    pause
    exit /b 1
)

echo.
echo ✓ 双模态训练完成！
echo 完成时间: %date% %time%
echo 模型保存在: checkpoints\stage1_timeseries_best.pth
echo.

REM ========================================================================
REM 步骤2: 训练语言模态版本
REM ========================================================================
echo.
echo [2/3] 训练语言模态版本 (预计 2-3 小时)...
echo ========================================================================
echo 开始时间: %date% %time%
echo.

python train_language_only.py > experiment_logs\step2_language_only_%timestamp%.log 2>&1

if %errorlevel% neq 0 (
    echo.
    echo ❌ 错误: 语言模态训练失败！
    echo 请查看日志: experiment_logs\step2_language_only_%timestamp%.log
    pause
    exit /b 1
)

echo.
echo ✓ 语言模态训练完成！
echo 完成时间: %date% %time%
echo 模型保存在: checkpoints\stage1_language_only_best.pth
echo.

REM ========================================================================
REM 步骤3: 对比结果
REM ========================================================================
echo.
echo [3/3] 对比两个版本的性能...
echo ========================================================================
echo.

python compare_results.py > experiment_logs\step3_comparison_%timestamp%.log 2>&1

if %errorlevel% neq 0 (
    echo.
    echo ⚠ 警告: 对比脚本执行失败
    echo 请查看日志: experiment_logs\step3_comparison_%timestamp%.log
    pause
    exit /b 1
)

REM 同时在终端显示对比结果
echo.
python compare_results.py

echo.
echo ========================================================================
echo ✓ 实验全部完成！
echo ========================================================================
echo 实验结束时间: %date% %time%
echo.
echo 结果文件:
echo   - checkpoints\stage1_timeseries_best.pth (双模态)
echo   - checkpoints\stage1_language_only_best.pth (语言模态)
echo   - comparison_results.json (对比结果)
echo.
echo 日志文件:
echo   - experiment_logs\step1_both_modal_%timestamp%.log
echo   - experiment_logs\step2_language_only_%timestamp%.log
echo   - experiment_logs\step3_comparison_%timestamp%.log
echo.
echo 查看训练曲线: tensorboard --logdir=logs
echo.

pause

