@echo off
echo ======================================================================
echo VTT 项目重组
echo ======================================================================
echo.
echo 此脚本将重组整个项目结构
echo.
echo 将执行以下操作:
echo   1. 创建新的目录结构（data/, docs/, archive/）
echo   2. 移动数据文件到 data/
echo   3. 整理文档到 docs/
echo   4. 归档旧脚本到 archive/
echo.
pause

echo.
echo 开始重组...
echo.

REM 1. 创建目录结构
echo [1/5] 创建目录结构...
if not exist "data" mkdir data
if not exist "docs" mkdir docs
if not exist "docs\guides" mkdir docs\guides
if not exist "docs\analysis" mkdir docs\analysis
if not exist "archive" mkdir archive
if not exist "archive\old_scripts" mkdir archive\old_scripts
if not exist "archive\old_docs" mkdir archive\old_docs

REM 2. 移动数据文件
echo [2/5] 移动数据文件...
if exist "extract2019_20251010_165007.csv" move "extract2019_20251010_165007.csv" "data\"
if exist "extract2020_20251010_165007.csv" move "extract2020_20251010_165007.csv" "data\"
if exist "extract2021_20251010_165007.csv" move "extract2021_20251010_165007.csv" "data\"
if exist "extract2022_20251010_165007.csv" move "extract2022_20251010_165007.csv" "data\"

REM 3. 整理文档
echo [3/5] 整理文档...

REM 主文档
if exist "EXPERIMENTS_README.md" move "EXPERIMENTS_README.md" "docs\EXPERIMENTS_GUIDE.md"
if exist "YIELD_PREDICTION_README.md" move "YIELD_PREDICTION_README.md" "docs\YIELD_PREDICTION_GUIDE.md"

REM 指南文档
if exist "QUICK_START.md" move "QUICK_START.md" "docs\guides\"
if exist "START_HERE.md" move "START_HERE.md" "docs\guides\"
if exist "TRAINING_GUIDE.md" move "TRAINING_GUIDE.md" "docs\guides\"
if exist "RUN_EXPERIMENT_GUIDE.md" move "RUN_EXPERIMENT_GUIDE.md" "docs\guides\"
if exist "TWO_STAGE_GUIDE.md" move "TWO_STAGE_GUIDE.md" "docs\guides\"
if exist "FLEXIBLE_CONFIG_GUIDE.md" move "FLEXIBLE_CONFIG_GUIDE.md" "docs\guides\"
if exist "VARIABLE_LENGTH_GUIDE.md" move "VARIABLE_LENGTH_GUIDE.md" "docs\guides\"

REM 分析文档
if exist "ALGORITHM_GUIDE.md" move "ALGORITHM_GUIDE.md" "docs\analysis\"
if exist "ARCHITECTURE_ANALYSIS.md" move "ARCHITECTURE_ANALYSIS.md" "docs\analysis\"
if exist "TWO_STAGE_VS_DIRECT_ANALYSIS.md" move "TWO_STAGE_VS_DIRECT_ANALYSIS.md" "docs\analysis\"
if exist "EXPERIMENT_DESIGN.md" move "EXPERIMENT_DESIGN.md" "docs\analysis\"
if exist "INPUT_MONTHS_COMPARISON.md" move "INPUT_MONTHS_COMPARISON.md" "docs\analysis\"

REM 归档文档
if exist "ABLATION_README.md" move "ABLATION_README.md" "archive\old_docs\"
if exist "ABLATION_EXPERIMENT_GUIDE.md" move "ABLATION_EXPERIMENT_GUIDE.md" "archive\old_docs\"
if exist "PROJECT_COMPLETE.md" move "PROJECT_COMPLETE.md" "archive\old_docs\"
if exist "PROJECT_SUMMARY.md" move "PROJECT_SUMMARY.md" "archive\old_docs\"
if exist "IMPLEMENTATION_SUMMARY.txt" move "IMPLEMENTATION_SUMMARY.txt" "archive\old_docs\"
if exist "EXPERIMENT_README.md" move "EXPERIMENT_README.md" "archive\old_docs\"

REM 4. 归档旧脚本
echo [4/5] 归档旧脚本...
if exist "train_language_only.py" move "train_language_only.py" "archive\old_scripts\"
if exist "train_multiyear_mirror.py" move "train_multiyear_mirror.py" "archive\old_scripts\"
if exist "run_ablation_experiment.py" move "run_ablation_experiment.py" "archive\old_scripts\"
if exist "run_ablation_simple.bat" move "run_ablation_simple.bat" "archive\old_scripts\"
if exist "run_language_only_quick.bat" move "run_language_only_quick.bat" "archive\old_scripts\"
if exist "run_ablation_experiment.bat" move "run_ablation_experiment.bat" "archive\old_scripts\"
if exist "run_all.bat" move "run_all.bat" "archive\old_scripts\"
if exist "test_language_only.py" move "test_language_only.py" "archive\old_scripts\"
if exist "compare_results.py" move "compare_results.py" "archive\old_scripts\"
if exist "compare_methods.py" move "compare_methods.py" "archive\old_scripts\"
if exist "quick_validation.py" move "quick_validation.py" "archive\old_scripts\"
if exist "run_full_experiment.py" move "run_full_experiment.py" "archive\old_scripts\"
if exist "visualize_results.py" move "visualize_results.py" "archive\old_scripts\"
if exist "train_variable_length.py" move "train_variable_length.py" "archive\old_scripts\"
if exist "test_variable_length.py" move "test_variable_length.py" "archive\old_scripts\"
if exist "data_loader_variable_length.py" move "data_loader_variable_length.py" "archive\old_scripts\"
if exist "data_loader_yield.py" move "data_loader_yield.py" "archive\old_scripts\"
if exist "quick_yield_test.py" move "quick_yield_test.py" "archive\old_scripts\"
if exist "train_yield_prediction.py" move "train_yield_prediction.py" "archive\old_scripts\"

REM 5. 清理临时文件
echo [5/5] 清理临时文件...
if exist "reorganize_project.py" del "reorganize_project.py"

echo.
echo ======================================================================
echo 重组完成！
echo ======================================================================
echo.
echo 新的项目结构:
echo   VTT/
echo   ├── data/                    # 数据文件
echo   ├── experiments/             # 实验代码
echo   ├── models/                  # 模型定义
echo   ├── docs/                    # 文档
echo   │   ├── guides/             # 使用指南
echo   │   └── analysis/           # 分析文档
echo   ├── archive/                 # 归档
echo   ├── checkpoints/            # 模型保存
echo   └── logs/                   # 训练日志
echo.
echo 下一步:
echo   1. 检查整理结果
echo   2. 运行快速实验: python experiments\yield_prediction\train.py --quick
echo.
pause

