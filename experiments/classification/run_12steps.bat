@echo off
chcp 65001 >nul
echo ====================================
echo 12步(120天)直接分类训练
echo ====================================
echo.
echo 请选择训练模式:
echo [1] 纯语言模态 (推荐先试，训练快)
echo [2] 真双模态 + 对比学习 (创新点，需预缓存图像)
echo.
set /p choice=请输入选项 (1 或 2): 

if "%choice%"=="1" goto language_only
if "%choice%"=="2" goto dual_modal

echo 无效选项
pause
exit

:language_only
echo.
echo ====================================
echo 纯语言模态训练
echo ====================================
echo 配置:
echo   - 时间步: 12步 (120天)
echo   - 模态: 纯语言 (CLIP-Text)
echo   - 训练速度: 快
echo   - 保存: timesclip_12steps_language_YYYYMMDD_HHMMSS/
echo.
pause
cd /d "%~dp0"
python train_12steps_language_only.py
goto end

:dual_modal
echo.
echo ====================================
echo 真双模态 + 对比学习训练 [创新点]
echo ====================================
echo 配置:
echo   - 时间步: 12步 (120天)
echo   - 模态: 视觉 + 语言
echo   - 对比学习: InfoNCE
echo   - 图像: 预缓存 (time_12/)
echo   - 保存: timesclip_12steps_dual_YYYYMMDD_HHMMSS/
echo.
echo 检查图像缓存...
if not exist "..\..\data\multiscale_image_cache\time_12" (
    echo 错误: 未找到图像缓存目录!
    echo 需要: data/multiscale_image_cache/time_12/
    pause
    exit
)
echo 图像缓存检查通过
echo.
echo 是否从上次中断处继续训练？
set /p resume_choice=[Y] 继续训练  [N] 全新开始 (默认N): 
echo.
cd /d "%~dp0"
if /i "%resume_choice%"=="y" (
    echo 正在从checkpoint恢复...
    python train_12steps_dual_cached.py --resume
) else (
    echo 开始全新训练...
    python train_12steps_dual_cached.py
)
goto end

:end
echo.
echo ====================================
echo 训练完成!
echo 请查看屏幕输出获取具体保存路径
echo ====================================
pause

