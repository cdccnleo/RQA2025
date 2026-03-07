@echo off
REM RQA2025 Docker镜像批量拉取脚本 (Windows)
REM Batch Pull Script for RQA2025 Docker Images

echo 🚀 RQA2025 Docker镜像批量拉取
echo =================================

REM 定义需要拉取的镜像列表
set IMAGES[0]=python:3.9-slim
set IMAGES[1]=postgres:15-alpine
set IMAGES[2]=redis:7-alpine
set IMAGES[3]=nginx:alpine
set IMAGES[4]=prom/prometheus:latest
set IMAGES[5]=grafana/grafana:latest

set TOTAL=6
set /a SUCCESS=0
set /a FAILED=0

echo 📦 需要拉取的镜像:
echo    - python:3.9-slim
echo    - postgres:15-alpine
echo    - redis:7-alpine
echo    - nginx:alpine
echo    - prom/prometheus:latest
echo    - grafana/grafana:latest
echo.

echo 🔄 开始拉取镜像...
echo.

for /L %%i in (0,1,5) do (
    setlocal enabledelayedexpansion
    set IMAGE=!IMAGES[%%i]!
    set /a COUNT=%%i+1

    echo [%%COUNT%/%TOTAL%] 拉取镜像: !IMAGE!
    docker pull !IMAGE!

    if !ERRORLEVEL! equ 0 (
        echo ✅ !IMAGE! 拉取成功
        set /a SUCCESS+=1
    ) else (
        echo ❌ !IMAGE! 拉取失败
        set /a FAILED+=1
    )
    echo.
    endlocal
)

echo 📊 拉取结果统计:
echo    总计镜像: %TOTAL%
echo    成功: %SUCCESS%
echo    失败: %FAILED%

if %FAILED% equ 0 (
    echo.
    echo 🎉 所有镜像拉取完成！
    echo.
    echo 🏗️ 接下来可以构建应用镜像:
    echo    docker build -t rqa2025:latest .
    echo.
    echo 🚀 启动服务:
    echo    docker-compose -f docker-compose.prod.yml up -d
) else (
    echo.
    echo ⚠️ 部分镜像拉取失败，请检查网络连接或尝试手动拉取失败的镜像
    echo.
    echo 🔄 重试命令:
    echo    docker pull python:3.9-slim
    echo    docker pull postgres:15-alpine
    echo    docker pull redis:7-alpine
    echo    docker pull nginx:alpine
    echo    docker pull prom/prometheus:latest
    echo    docker pull grafana/grafana:latest
)

echo.
pause


