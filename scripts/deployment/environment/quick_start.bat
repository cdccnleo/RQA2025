@echo off
chcp 6501echo RQA2025快速启动脚本
echo ========================================
echo.

:: 检查是否在正确的目录
if not existsrc" (
    echo 错误: 请在项目根目录下运行此脚本
    pause
    exit /b 1
:: 检查conda环境
echo1 检查conda环境...
call conda info --envs | findstr "rqa"
if %errorlevel% neq 0(
    echo 警告: 未找到rqa环境
    echo 正在创建rqa环境...
    call conda create -n rqa python=3.9    if %errorlevel% neq0 (
        echo 错误: 无法创建rqa环境
        pause
        exit /b1
    )
)

:: 激活rqa环境
echo [2rqa环境...
call conda activate rqa
if %errorlevel% neq0 (
    echo 错误: 无法激活rqa环境
    pause
    exit /b 1
)

:: 检查并安装依赖
echo [3] 检查依赖...
python -m pytest --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装pytest...
    pip install pytest pytest-cov
)

:: 创建报告目录
if not exist "reports mkdir reports
if not existdocs" mkdir docs

:: 运行测试
echo 4 运行分层测试...
echo.

:: 运行基础设施层测试
echo 执行基础设施层测试...
python -m pytest tests/unit/infrastructure/ --cov=src/infrastructure --cov-report=html:reports/infrastructure_coverage.html --cov-report=term-missing -v --tb=short
set INFRA_RESULT=%errorlevel%

:: 运行数据层测试
echo.
echo 执行数据层测试...
python -m pytest tests/unit/data/ --cov=src/data --cov-report=html:reports/data_coverage.html --cov-report=term-missing -v --tb=short
set DATA_RESULT=%errorlevel%

:: 运行特征层测试
echo.
echo 执行特征层测试...
python -m pytest tests/unit/features/ --cov=src/features --cov-report=html:reports/features_coverage.html --cov-report=term-missing -v --tb=short
set FEATURES_RESULT=%errorlevel%

:: 运行模型层测试
echo.
echo 执行模型层测试...
python -m pytest tests/unit/models/ --cov=src/models --cov-report=html:reports/models_coverage.html --cov-report=term-missing -v --tb=short
set MODELS_RESULT=%errorlevel%

:: 运行交易层测试
echo.
echo 执行交易层测试...
python -m pytest tests/unit/trading/ --cov=src/trading --cov-report=html:reports/trading_coverage.html --cov-report=term-missing -v --tb=short
set TRADING_RESULT=%errorlevel%

:: 运行回测层测试
echo.
echo 执行回测层测试...
python -m pytest tests/unit/backtest/ --cov=src/backtest --cov-report=html:reports/backtest_coverage.html --cov-report=term-missing -v --tb=short
set BACKTEST_RESULT=%errorlevel%

:: 显示结果
echo.
echo ========================================
echo 测试执行结果摘要
echo ========================================
echo 基础设施层: %INFRA_RESULT% (0成功, 非0=失败)
echo 数据层: %DATA_RESULT% (0成功, 非0=失败)
echo 特征层: %FEATURES_RESULT% (0成功, 非0=失败)
echo 模型层: %MODELS_RESULT% (0成功, 非0=失败)
echo 交易层: %TRADING_RESULT% (0=成功, 非0=失败)
echo 回测层: %BACKTEST_RESULT% (0=成功, 非0失败)
echo.

:: 计算总体结果
set /a TOTAL_RESULT=%INFRA_RESULT%+%DATA_RESULT%+%FEATURES_RESULT%+%MODELS_RESULT%+%TRADING_RESULT%+%BACKTEST_RESULT%

if %TOTAL_RESULT% equ 0(
    echo ✅ 所有层测试执行成功！
) else (
    echo ⚠️  部分层测试存在问题，请查看详细报告
)

echo.
echo 报告文件位置:
echo - 覆盖率报告: reports/
echo - 使用指南: docs/conda_environment_guide.md
echo.

echo 快速启动完成！
pause 