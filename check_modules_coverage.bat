@echo off
echo 🚀 基础设施层16个子模块测试覆盖率检查
echo =====================================
echo.

cd /d C:\PythonProject\RQA2025

set "modules=config distributed versioning resource logging ops monitoring health security constants interfaces optimization core utils cache error"

set total_modules=0
set passed_modules=0

for %%m in (%modules%) do (
    set /a total_modules+=1
    echo.
    echo 🔍 检查模块: %%m
    echo ──────────────────

    python -m pytest --cov=src/infrastructure/%%m --cov-report=term --cov-fail-under=0 tests/unit/infrastructure/%%m/ -q --tb=no 2>nul | findstr "TOTAL" > temp_coverage.txt 2>nul

    if exist temp_coverage.txt (
        for /f "tokens=*" %%i in (temp_coverage.txt) do (
            echo %%i
            for /f "tokens=4" %%c in ("%%i") do (
                set "coverage=%%c"
                if !coverage! geq 80 (
                    echo ✅ 达标 (80%%+)
                    set /a passed_modules+=1
                ) else (
                    echo ❌ 未达标
                )
            )
        )
        del temp_coverage.txt
    ) else (
        echo ❌ 无覆盖率数据或测试失败
    )
)

echo.
echo 📊 总结报告
echo ==========
echo 检查模块总数: %total_modules%
echo 达标模块数量: %passed_modules%
set /a not_passed=%total_modules%-%passed_modules%
echo 未达标模块数: %not_passed%

if %passed_modules% geq 12 (
    echo.
    echo 🎉 基础设施层测试覆盖率已达标投产要求！
) else (
    echo.
    echo ⚠️ 基础设施层测试覆盖率未达标，需要继续改进
)

echo.
pause

























