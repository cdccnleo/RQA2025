@echo off
REM RQA2025 Docker部署批处理脚本
REM Automated Docker Deployment Script for RQA2025

echo 🚀 RQA2025 量化交易系统 - Docker容器部署
echo =================================================

cd /d "%~dp0"

echo 🔧 检查环境...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Python未安装或不在PATH中
    pause
    exit /b 1
)

echo ✅ Python环境正常

echo 🐳 开始Docker部署...
python deploy_production.py docker

if %ERRORLEVEL% equ 0 (
    echo.
    echo 🎉 部署成功!
    echo 🌐 服务访问地址:
    echo    主应用: http://localhost:8000
    echo    API文档: http://localhost:8000/docs
    echo    Grafana: http://localhost:3000
    echo.
    echo 📊 查看服务状态: docker-compose -f docker-compose.prod.yml ps
    echo 📋 查看日志: docker-compose -f docker-compose.prod.yml logs -f
) else (
    echo.
    echo ❌ 部署失败，请检查上述错误信息
)

echo.
pause


