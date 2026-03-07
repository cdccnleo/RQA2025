@echo off
chcp65001=========================
echo RQA2025分层测试自动化脚本 (Conda RQA环境)
echo ========================================
echo.

:: 检查conda环境
echo [1 检查conda环境...
call conda info --envs | findstr "rqa"
if %errorlevel% neq 0(
    echo 错误: 未找到rqa环境，请先创建并激活rqa环境
    echo 运行命令: conda create -n rqa python=3.9   echo 然后激活: conda activate rqa
    pause
    exit /b 1
)

:: 激活rqa环境
echo [2/6rqa环境...
call conda activate rqa
if %errorlevel% neq0 (
    echo 错误: 无法激活rqa环境
    pause
    exit /b 1:: 检查Python和pytest
echo [3/6查Python和pytest...
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未正确安装或激活
    pause
    exit /b 1
)

pytest --version
if %errorlevel% neq 0 (
    echo 错误: pytest未安装，正在安装...
    pip install pytest pytest-cov
)

:: 设置环境变量
echo [4] 设置环境变量...
set PYTHONPATH=%cd%
set RQA_ENV=conda_rqa
set TEST_MODE=layered

:: 创建报告目录
if not exist "reports mkdir reports
if not existdocsmkdir docs

echo [5/6] 开始分层测试执行...
echo.

:: 基础设施层测试
echo ========================================
echo 执行基础设施层测试...
echo ========================================
pytest tests/unit/infrastructure/ --cov=src/infrastructure --cov-report=html:reports/infrastructure_coverage.html --cov-report=term-missing --cov-report=json:reports/infrastructure_coverage.json -v --tb=short
set INFRA_RESULT=%errorlevel%

:: 数据层测试
echo ========================================
echo 执行数据层测试...
echo ========================================
pytest tests/unit/data/ --cov=src/data --cov-report=html:reports/data_coverage.html --cov-report=term-missing --cov-report=json:reports/data_coverage.json -v --tb=short
set DATA_RESULT=%errorlevel%

:: 特征层测试
echo ========================================
echo 执行特征层测试...
echo ========================================
pytest tests/unit/features/ --cov=src/features --cov-report=html:reports/features_coverage.html --cov-report=term-missing --cov-report=json:reports/features_coverage.json -v --tb=short
set FEATURES_RESULT=%errorlevel%

:: 模型层测试
echo ========================================
echo 执行模型层测试...
echo ========================================
pytest tests/unit/models/ --cov=src/models --cov-report=html:reports/models_coverage.html --cov-report=term-missing --cov-report=json:reports/models_coverage.json -v --tb=short
set MODELS_RESULT=%errorlevel%

:: 交易层测试
echo ========================================
echo 执行交易层测试...
echo ========================================
pytest tests/unit/trading/ --cov=src/trading --cov-report=html:reports/trading_coverage.html --cov-report=term-missing --cov-report=json:reports/trading_coverage.json -v --tb=short
set TRADING_RESULT=%errorlevel%

:: 回测层测试
echo ========================================
echo 执行回测层测试...
echo ========================================
pytest tests/unit/backtest/ --cov=src/backtest --cov-report=html:reports/backtest_coverage.html --cov-report=term-missing --cov-report=json:reports/backtest_coverage.json -v --tb=short
set BACKTEST_RESULT=%errorlevel%

echo [6/6] 生成汇总报告...
echo.

:: 运行自动化脚本
echo 运行自动化推进脚本...
python scripts/auto_model_landing.py

echo 运行技术债务管理脚本...
python scripts/technical_debt_manager.py

echo 运行进度跟踪脚本...
python scripts/progress_tracker.py

:: 显示结果摘要
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
echo - 进度报告: docs/progress_report.md
echo - 技术债务: docs/technical_debt_report.md
echo.

echo 脚本执行完成！
pause 