@echo off
echo ======================================================================
echo 对比实验: 语言模态 vs 双模态产量预测
echo ======================================================================
echo.
echo 这个实验将同时训练两种模型并对比性能:
echo   1. 语言模态only (LanguageOnlyYieldPredictor)
echo   2. 双模态 (SimpleYieldPredictor with Vision)
echo.
echo 选择运行模式:
echo   1. 快速测试 (1-2小时, 测试4个时间点: 60,120,180,300天)
echo   2. 完整实验 (4-6小时, 测试6个时间点: 60,120,180,240,300,360天)
echo   3. 自定义轮数
echo.
set /p choice=请输入选项 (1/2/3): 

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto full
if "%choice%"=="3" goto custom
goto invalid

:quick
echo.
echo 启动快速对比测试...
echo   - 测试点: 60天, 120天, 180天, 300天
echo   - 每个模型训练30轮
echo   - 预计耗时: 1-2小时
echo.
pause
python experiments/yield_prediction/train_comparison.py --quick
goto end

:full
echo.
echo 启动完整对比实验...
echo   - 测试点: 60天, 120天, 180天, 240天, 300天, 360天
echo   - 每个模型训练50轮
echo   - 预计耗时: 4-6小时
echo.
pause
python experiments/yield_prediction/train_comparison.py
goto end

:custom
echo.
set /p epochs=请输入训练轮数 (建议30-100): 
echo.
echo 启动完整对比实验 (自定义轮数)...
echo   - 测试点: 60天, 120天, 180天, 240天, 300天, 360天
echo   - 每个模型训练%epochs%轮
echo.
pause
python experiments/yield_prediction/train_comparison.py --epochs %epochs%
goto end

:invalid
echo.
echo 无效选项！请重新运行脚本。
pause
goto end

:end
echo.
echo ======================================================================
echo 实验完成！
echo ======================================================================
echo.
echo 查看结果:
echo   - 对比JSON: experiments\yield_prediction\comparison\results\comparison.json
echo   - 对比图表: experiments\yield_prediction\comparison\results\comparison.png
echo   - 训练曲线: tensorboard --logdir=experiments/yield_prediction/comparison/logs
echo.
echo 图表说明:
echo   - 左上: RMSE对比 (越低越好)
echo   - 右上: R²对比 (越高越好)
echo   - 左下: MAE对比 (越低越好)
echo   - 右下: RMSE差异 (负值=语言更好, 正值=双模态更好)
echo.
pause

