@echo off
chcp 65001 >nul
echo.
echo ========================================
echo TimesCLIP 分类任务训练脚本
echo ========================================
echo.
echo 请选择要训练的模型：
echo [1] 双模态模型（TimesCLIP完整版）
echo [2] 纯语言模型（仅CLIP-Text）
echo [3] 训练两个模型并对比
echo [4] 仅对比已有模型结果
echo [0] 退出
echo.
set /p choice="请输入选项 [0-4]: "

if "%choice%"=="1" goto train_dual
if "%choice%"=="2" goto train_language
if "%choice%"=="3" goto train_both
if "%choice%"=="4" goto compare
if "%choice%"=="0" goto end

echo 无效选项，请重新运行
goto end

:train_dual
echo.
echo ========================================
echo 训练双模态模型
echo ========================================
python experiments/classification/train_classification_timesclip.py --model_type dual --epochs 100
goto end

:train_language
echo.
echo ========================================
echo 训练纯语言模型
echo ========================================
python experiments/classification/train_classification_timesclip.py --model_type language_only --epochs 100
goto end

:train_both
echo.
echo ========================================
echo 训练两个模型
echo ========================================
echo.
echo [1/2] 训练双模态模型...
python experiments/classification/train_classification_timesclip.py --model_type dual --epochs 100
echo.
echo [2/2] 训练纯语言模型...
python experiments/classification/train_classification_timesclip.py --model_type language_only --epochs 100
echo.
echo ========================================
echo 对比两个模型
echo ========================================
python experiments/classification/compare_classification_models.py
goto end

:compare
echo.
echo ========================================
echo 对比已有模型结果
echo ========================================
python experiments/classification/compare_classification_models.py
goto end

:end
echo.
pause

