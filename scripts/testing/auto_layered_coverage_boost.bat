@echo off
setlocal enabledelayedexpansion

REM 一键分层自动循环推进测试覆盖率提升
REM 可根据需要修改循环轮数
set ROUNDS=3
set LAYERS=infrastructure data features models trading backtest

for /l %%R in (1,1,%ROUNDS%) do (
    echo ===============================
    echo 第 %%R 轮自动化分层推进开始
    echo ===============================
    for %%L in (%LAYERS%) do (
        echo -------------------------------
        echo [%%R] 正在推进 %%L 层 ...
        echo -------------------------------
        python scripts/testing/enhance_test_coverage_plan.py --target 80 --phase all --layer %%L
        echo.
    )
    echo ===============================
    echo 第 %%R 轮自动化分层推进结束
    echo ===============================
)

echo 全部分层自动化推进已完成！
pause 