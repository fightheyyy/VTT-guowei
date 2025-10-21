@echo off
REM Windows批处理脚本 - 一键训练

echo ============================================================
echo TimesCLIP 一键训练脚本
echo ============================================================

echo.
echo [步骤 1/3] 测试数据加载...
python test_data_loading.py
if errorlevel 1 (
    echo 数据加载失败！请检查数据文件。
    pause
    exit /b 1
)

echo.
echo 数据测试通过！
pause

echo.
echo [步骤 2/3] 开始训练（这将需要1-2小时）...
echo 您可以按 Ctrl+C 取消训练
python train_two_stage.py
if errorlevel 1 (
    echo 训练失败！
    pause
    exit /b 1
)

echo.
echo [步骤 3/3] 使用模拟数据测试预测...
python predict_2025.py

echo.
echo ============================================================
echo 训练完成！
echo ============================================================
echo.
echo 模型文件保存在:
echo   - checkpoints/stage1_timeseries_best.pth
echo   - checkpoints/stage2_yield_best.pth
echo.
echo 现在可以使用真实的2025年数据进行预测了！
echo.
pause

