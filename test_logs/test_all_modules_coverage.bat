@echo off
REM 基础设施层所有模块真实覆盖率测试脚本
setlocal enabledelayedexpansion

echo ========================================
echo 基础设施层真实覆盖率测试
echo ========================================
echo.

set "REPORT_FILE=test_logs\real_coverage_results.txt"
echo 生成时间: %date% %time% > %REPORT_FILE%
echo. >> %REPORT_FILE%
echo 模块覆盖率测试结果 >> %REPORT_FILE%
echo ======================================== >> %REPORT_FILE%
echo. >> %REPORT_FILE%

REM 定义模块列表
set MODULES=constants interfaces core ops optimization distributed versioning api error monitoring cache security logging resource health config utils

for %%M in (%MODULES%) do (
    echo.
    echo [%%M] 测试中...
    echo [%%M] >> %REPORT_FILE%
    
    python -m pytest tests\unit\infrastructure\%%M --cov=src\infrastructure\%%M --cov-report=term -q --tb=no --no-header --maxfail=1 2>&1 | findstr /C:"TOTAL" >> %REPORT_FILE%
    
    if !ERRORLEVEL! EQU 0 (
        echo   完成
    ) else (
        echo   执行出错
    )
)

echo.
echo ========================================
echo 测试完成！查看报告:
type %REPORT_FILE%
echo.
echo 报告已保存至: %REPORT_FILE%
pause

