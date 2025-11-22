@echo off
chcp 65001 >nul
echo ====================================
echo 恢复昨天的训练
echo ====================================
echo.
echo 昨天训练信息:
echo   - 目录: timesclip_12steps_dual_20251120_221120
echo   - 最佳Val F1: 0.5626 (Epoch 15)
echo   - 已完成: 15/100 epochs
echo   - 剩余: 85 epochs
echo.
echo 选择操作:
echo [1] 从Epoch 16继续训练（断点续训）
echo [2] 评估最佳模型（在测试集上）
echo [3] 查看训练历史
echo.
set /p choice=请输入选项 (1/2/3): 

if "%choice%"=="1" goto resume_training
if "%choice%"=="2" goto evaluate
if "%choice%"=="3" goto view_history

echo 无效选项
pause
exit

:resume_training
echo.
echo 正在恢复训练...
cd /d "%~dp0"
python train_12steps_dual_cached.py --resume
goto end

:evaluate
echo.
echo [功能开发中] 评估最佳模型
echo.
echo 最佳模型路径:
echo experiments/classification/timesclip_12steps_dual_20251120_221120/checkpoints/best_model.pth
echo.
echo 可以手动加载此模型进行评估
pause
goto end

:view_history
echo.
echo [功能开发中] 查看训练历史
echo.
echo 配置文件:
type experiments\classification\timesclip_12steps_dual_20251120_221120\config.json
echo.
pause
goto end

:end
pause

