@echo off
REM RQA2025 快速启动批处理脚本
REM Quick Start Batch Script for RQA2025

echo 🚀 RQA2025 量化交易系统 - 快速启动
echo ==================================================
echo.

echo 📋 检查系统要求...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或不在PATH中
    echo 请安装Python 3.9+并添加到PATH
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if (sys.version_info.major == 3 and sys.version_info.minor >= 9) else 1)"
if %errorlevel% neq 0 (
    echo ❌ Python版本需要 3.9+
    python --version
    pause
    exit /b 1
)

echo ✅ Python版本检查通过
echo.

echo 请选择启动方式:
echo 1. Docker生产环境启动 (推荐)
echo 2. 本地Python环境启动
echo 3. 运行快速启动脚本
echo 4. 退出
echo.

:menu
set /p choice="请输入选择 (1-4): "

if "%choice%"=="1" goto docker_start
if "%choice%"=="2" goto local_start
if "%choice%"=="3" goto script_start
if "%choice%"=="4" goto exit_app

echo ❌ 无效选择，请重新输入
goto menu

:docker_start
echo.
echo 🐳 Docker生产环境启动
echo ------------------------------
echo.

docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker未安装或未运行
    echo 请安装并启动Docker Desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose未安装
    pause
    exit /b 1
)

echo ✅ Docker环境检查通过
echo.

echo 🔄 启动RQA2025服务...
docker-compose -f docker-compose.prod.yml up -d

if %errorlevel% neq 0 (
    echo ❌ 服务启动失败
    pause
    exit /b 1
)

echo ✅ 服务启动命令执行成功
echo.
echo ⏳ 等待服务就绪...

REM 等待服务启动
timeout /t 10 /nobreak >nul

echo 🌐 服务访问信息:
echo    主应用:    http://localhost:8000
echo    API文档:   http://localhost:8000/docs
echo    健康检查:  http://localhost:8000/health
echo    Grafana:   http://localhost:3000
echo    Prometheus: http://localhost:9090
echo.

echo 📊 查看服务状态:
docker-compose -f docker-compose.prod.yml ps
echo.

echo 🎉 RQA2025系统启动成功!
echo 按任意键查看日志，或关闭此窗口保持服务运行...
pause >nul

REM 显示日志
docker-compose -f docker-compose.prod.yml logs -f app
goto end

:local_start
echo.
echo 🐍 本地Python环境启动
echo ------------------------------
echo.

conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda未安装
    echo 请安装Miniconda或Anaconda
    pause
    exit /b 1
)

echo ✅ Conda环境检查通过
echo.

echo 🔧 检查rqa环境...
conda info --envs | findstr "rqa" >nul
if %errorlevel% neq 0 (
    echo ⚠️  rqa环境不存在，正在创建...
    conda create -n rqa python=3.9 -y
    if %errorlevel% neq 0 (
        echo ❌ 创建conda环境失败
        pause
        exit /b 1
    )
)

echo ✅ conda环境准备就绪
echo.

echo 📦 安装依赖...
conda run -n rqa pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

echo ✅ 依赖安装完成
echo.

echo 🚀 启动服务...
echo 服务将在新窗口中启动...
start "RQA2025 Service" cmd /k "conda activate rqa && python main.py"

echo ⏳ 等待服务启动...
timeout /t 5 /nobreak >nul

echo 🌐 服务访问信息:
echo    主应用:    http://localhost:8000
echo    API文档:   http://localhost:8000/docs
echo    健康检查:  http://localhost:8000/health
echo.

echo 🎉 RQA2025系统启动成功!
echo 按任意键继续...
pause >nul
goto end

:script_start
echo.
echo 🐍 运行Python快速启动脚本
echo ------------------------------
python quick_start.py
goto end

:exit_app
echo 👋 再见!
goto end

:end


