@echo off
chcp 65001 >nul
echo ========================================
echo TimesCLIP产量预测训练
echo ========================================
echo.

:menu
echo 请选择实验：
echo.
echo [1] 快速测试（10 epochs，12步）
echo [2] 完整TimesCLIP（双模态+对比学习+变量选择）
echo [3] 纯语言模态（CLIP-Text only）
echo [4] 不使用对比学习
echo [5] 不使用变量选择
echo [6] 对比原始方法 vs TimesCLIP
echo [7] 消融实验（完整）
echo [8] 退出
echo.
set /p choice="请输入选项 [1-8]: "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto full_timesclip
if "%choice%"=="3" goto language_only
if "%choice%"=="4" goto no_contrastive
if "%choice%"=="5" goto no_variate
if "%choice%"=="6" goto comparison
if "%choice%"=="7" goto ablation
if "%choice%"=="8" goto end
echo 无效选项，请重新选择
goto menu

:quick_test
echo.
echo ========================================
echo 快速测试
echo ========================================
python experiments/yield_prediction/train_timesclip.py --quick --input_steps 12
echo.
pause
goto menu

:full_timesclip
echo.
echo ========================================
echo 完整TimesCLIP训练
echo ========================================
echo 包含：CLIP-Text + CLIP-Vision + 对比学习 + 变量选择
echo.
set /p steps="输入时间步数 (6-36, 默认12): "
if "%steps%"=="" set steps=12
echo.
python experiments/yield_prediction/train_timesclip.py --input_steps %steps% --epochs 100
echo.
pause
goto menu

:language_only
echo.
echo ========================================
echo 纯语言模态（CLIP-Text only）
echo ========================================
set /p steps="输入时间步数 (6-36, 默认12): "
if "%steps%"=="" set steps=12
echo.
python experiments/yield_prediction/train_timesclip.py --language_only --input_steps %steps% --epochs 100
echo.
pause
goto menu

:no_contrastive
echo.
echo ========================================
echo 不使用对比学习
echo ========================================
set /p steps="输入时间步数 (6-36, 默认12): "
if "%steps%"=="" set steps=12
echo.
python experiments/yield_prediction/train_timesclip.py --no_contrastive --input_steps %steps% --epochs 100
echo.
pause
goto menu

:no_variate
echo.
echo ========================================
echo 不使用变量选择
echo ========================================
set /p steps="输入时间步数 (6-36, 默认12): "
if "%steps%"=="" set steps=12
echo.
python experiments/yield_prediction/train_timesclip.py --no_variate_selection --input_steps %steps% --epochs 100
echo.
pause
goto menu

:comparison
echo.
echo ========================================
echo 对比实验：原始方法 vs TimesCLIP
echo ========================================
echo.
echo 将训练以下模型：
echo 1. 原始双模态（从头训练的Transformer）
echo 2. TimesCLIP完整版（CLIP-Text + 对比学习 + 变量选择）
echo.
set /p confirm="确认开始对比实验？(y/n): "
if /i not "%confirm%"=="y" goto menu

echo.
echo [1/2] 训练原始方法...
python experiments/yield_prediction/train_comparison.py --quick
echo.
echo [2/2] 训练TimesCLIP...
python experiments/yield_prediction/train_timesclip.py --quick
echo.
echo 对比实验完成！
pause
goto menu

:ablation
echo.
echo ========================================
echo 消融实验
echo ========================================
echo.
echo 将训练以下配置：
echo 1. 完整TimesCLIP
echo 2. 不使用对比学习
echo 3. 不使用变量选择
echo 4. 纯语言模态
echo.
set /p confirm="确认开始消融实验？(y/n): "
if /i not "%confirm%"=="y" goto menu

set steps=12
set epochs=50

echo.
echo [1/4] 完整TimesCLIP...
python experiments/yield_prediction/train_timesclip.py --input_steps %steps% --epochs %epochs%

echo.
echo [2/4] 不使用对比学习...
python experiments/yield_prediction/train_timesclip.py --no_contrastive --input_steps %steps% --epochs %epochs%

echo.
echo [3/4] 不使用变量选择...
python experiments/yield_prediction/train_timesclip.py --no_variate_selection --input_steps %steps% --epochs %epochs%

echo.
echo [4/4] 纯语言模态...
python experiments/yield_prediction/train_timesclip.py --language_only --input_steps %steps% --epochs %epochs%

echo.
echo 消融实验完成！
pause
goto menu

:end
echo.
echo 再见！
exit /b 0

