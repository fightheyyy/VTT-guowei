@echo off
chcp 65001 >nul
echo ====================================
echo 改进版训练 - 处理类别不平衡
echo ====================================
echo.

:menu
echo 请选择训练模式:
echo [1] 端到端Pipeline (推荐)
echo [2] 仅Forecaster
echo [3] 继续训练 (从checkpoint恢复)
echo [4] 对比实验 (原版 vs 改进版)
echo [0] 退出
echo.
set /p choice=请输入选项 (0-4): 

if "%choice%"=="1" goto pipeline
if "%choice%"=="2" goto forecaster
if "%choice%"=="3" goto resume
if "%choice%"=="4" goto compare
if "%choice%"=="0" goto end

echo 无效选项，请重试
goto menu

:pipeline
echo.
echo === 训练端到端Pipeline (改进版) ===
echo 特性:
echo   - Focal Loss (gamma=2.0)
echo   - 类别权重 (自动计算)
echo   - 高dropout (0.3)
echo   - 低学习率 (5e-5)
echo   - 梯度裁剪
echo.
python train_pipeline_improved.py ^
    --csv_path ../../data/2018four.csv ^
    --input_len 6 ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 5e-5 ^
    --dropout 0.3 ^
    --focal_gamma 2.0
echo.
pause
goto menu

:forecaster
echo.
echo === 训练Forecaster (改进版) ===
echo 特性:
echo   - 数据增强
echo   - 高正则化
echo   - 平衡采样 (可选)
echo.
python train_forecaster_improved.py ^
    --csv_path ../../data/2018four.csv ^
    --input_len 6 ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 5e-5 ^
    --dropout 0.3 ^
    --weight_decay 1e-3 ^
    --use_augmentation
echo.
pause
goto menu

:resume
echo.
echo === 从checkpoint继续训练 ===
echo.
set /p ckpt=请输入checkpoint路径 (如: checkpoints/pipeline_e2e_in6.pth): 
if not exist "%ckpt%" (
    echo 错误: checkpoint不存在
    pause
    goto menu
)
echo.
echo 继续训练: %ckpt%
python train_pipeline_improved.py ^
    --checkpoint %ckpt% ^
    --epochs 100 ^
    --lr 5e-5
echo.
pause
goto menu

:compare
echo.
echo === 对比实验: 原版 vs 改进版 ===
echo.
echo [1/2] 训练原版...
python two_stage_pipeline.py ^
    --stage e2e ^
    --input_len 6 ^
    --epochs 50
echo.
echo [2/2] 训练改进版...
python train_pipeline_improved.py ^
    --input_len 6 ^
    --epochs 50
echo.
echo 对比完成! 查看checkpoints目录对比结果
pause
goto menu

:end
echo 退出

