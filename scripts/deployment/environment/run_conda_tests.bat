@echo off
chcp 65001
echo RQA2025 Conda环境测试运行器
echo ========================================
echo.

:: 检查conda环境
echo [1] 检查conda环境...
conda info --envs | findstr "test"
if %errorlevel% neq 0 (
    echo 错误: 未找到test环境，请先创建并激活test环境
    echo 运行命令: conda create -n test python=3.9
    pause
    exit /b 1
)

:: 检查Python和pytest
echo [2] 检查Python和pytest...
conda run -n test python --version
if %errorlevel% neq 0 (
    echo 错误: Python未正确安装
    pause
    exit /b 1
)

conda run -n test python -m pytest --version
if %errorlevel% neq 0 (
    echo 警告: pytest未安装，正在安装...
    conda run -n test pip install pytest pytest-cov
)

:: 设置环境变量
echo [3] 设置环境变量...
set PYTHONPATH=%cd%
set RQA_ENV=conda_test
set TEST_MODE=layered

:: 创建报告目录
if not exist "reports" mkdir reports
if not exist "docs" mkdir docs

echo [4] 开始分层测试执行...
echo.

:: 定义测试层
set LAYERS=infrastructure data features models trading backtest

:: 运行测试
for %%l in (%LAYERS%) do (
    echo ========================================
    echo 执行%%l层测试...
    echo ========================================
    conda run -n test python -m pytest tests/unit/%%l/ --cov=src/%%l --cov-report=html:reports/%%l_coverage.html --cov-report=term-missing --cov-report=json:reports/%%l_coverage.json -v --tb=short
    if !errorlevel! equ 0 (
        echo ✓ %%l层测试成功
    ) else (
        echo ✗ %%l层测试失败
    )
    echo.
)

:: 运行自动化推进脚本
echo 运行自动化推进脚本...
conda run -n test python scripts/conda_test_runner.py

echo ========================================
echo 测试执行完成！
echo ========================================
echo.
echo 报告文件位置:
echo - 覆盖率报告: reports/
echo - 测试报告: docs/conda_test_report.md
echo - 测试数据: docs/conda_test_results.json
echo.
pause 