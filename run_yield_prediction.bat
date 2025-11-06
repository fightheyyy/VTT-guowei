@echo off
echo ======================================================================
echo VTT - 产量预测实验
echo ======================================================================
echo.
echo 这个实验将测试不同输入长度对产量预测准确度的影响
echo.
echo 选择运行模式:
echo   1. 快速测试 (1-2小时, 测试4个时间点)
echo   2. 完整实验 (4-8小时, 测试12个时间点)
echo   3. 自定义轮数
echo   4. 查看帮助
echo.
set /p choice=请输入选项 (1/2/3/4): 

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto full
if "%choice%"=="3" goto custom
if "%choice%"=="4" goto help
goto invalid

:quick
echo.
echo 启动快速测试模式...
echo   - 测试点: 60天, 120天, 180天, 300天
echo   - 训练轮数: 30轮/模型
echo   - 预计耗时: 1-2小时
echo.
pause
python experiments/yield_prediction/train.py --quick
goto end

:full
echo.
echo 启动完整实验模式...
echo   - 测试点: 30天到360天，间隔30天 (共12个点)
echo   - 训练轮数: 50轮/模型
echo   - 预计耗时: 4-8小时
echo.
pause
python experiments/yield_prediction/train.py
goto end

:custom
echo.
set /p epochs=请输入训练轮数 (建议30-100): 
echo.
echo 启动完整实验模式 (自定义轮数)...
echo   - 测试点: 30天到360天，间隔30天 (共12个点)
echo   - 训练轮数: %epochs%轮/模型
echo.
pause
python experiments/yield_prediction/train.py --epochs %epochs%
goto end

:help
echo.
echo 产量预测实验说明
echo ======================================================================
echo.
echo 实验目标:
echo   找到最短的有效预测天数，用于农作物产量预测
echo.
echo 数据说明:
echo   - 训练集: 2019-2021年 (约1500样本)
echo   - 测试集: 2022年 (约500样本)
echo   - 波段: 7个主要遥感指标 (NIR, RVI, SWIR1, blue, evi, ndvi, red)
echo   - 时间步: 36步 (每步10天 = 360天 = 1年)
echo.
echo 输出结果:
echo   - results.json       结果JSON
echo   - analysis.png       性能对比图表
echo   - checkpoints/       训练好的模型
echo   - logs/              TensorBoard日志
echo.
echo 查看结果:
echo   tensorboard --logdir=experiments/yield_prediction/logs
echo.
echo 按任意键返回...
pause > nul
cls
goto :eof

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
echo   - 结果JSON: experiments\yield_prediction\results\results.json
echo   - 性能图表: experiments\yield_prediction\results\analysis.png
echo   - 训练曲线: tensorboard --logdir=experiments/yield_prediction/logs
echo.
pause

