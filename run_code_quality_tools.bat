@echo off
chcp 65001 >nul
echo ==========================================
echo RQA2025 代码质量工具执行脚本
echo ==========================================
echo.

REM 设置Python环境
set PYTHONHOME=
set PYTHONPATH=

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请确保Python已安装并添加到PATH
    exit /b 1
)

echo [1/5] 正在运行Black代码格式化...
python -m black src --line-length 100 --target-version py39
if errorlevel 1 (
    echo [警告] Black执行失败，继续执行其他工具...
) else (
    echo [✓] Black格式化完成
)
echo.

echo [2/5] 正在运行isort导入排序...
python -m isort src --profile black --line-length 100
if errorlevel 1 (
    echo [警告] isort执行失败，继续执行其他工具...
) else (
    echo [✓] isort排序完成
)
echo.

echo [3/5] 正在运行Flake8代码检查...
python -m flake8 src --max-line-length=100 --extend-ignore=E203,W503 --count --statistics
if errorlevel 1 (
    echo [信息] Flake8发现一些问题，请查看上方输出
) else (
    echo [✓] Flake8检查通过
)
echo.

echo [4/5] 正在执行批量修复脚本...
if exist "scripts\batch_fix_simple_issues.py" (
    python scripts\batch_fix_simple_issues.py
    echo [✓] 批量修复完成
) else (
    echo [警告] 批量修复脚本不存在
)
echo.

echo [5/5] 正在生成质量报告...
if exist "scripts\generate_quality_report.py" (
    python scripts\generate_quality_report.py
    echo [✓] 质量报告生成完成
) else (
    echo [警告] 质量报告脚本不存在
)
echo.

echo ==========================================
echo 代码质量工具执行完成！
echo ==========================================
echo.
echo 下一步建议：
echo 1. 查看生成的质量报告
echo 2. 检查修改的文件：git status
echo 3. 提交更改：git add -A ^&^& git commit -m "style: format code"
echo 4. 推送到GitHub：git push origin main
echo.
pause
