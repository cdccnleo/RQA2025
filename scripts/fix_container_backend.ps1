# 修复容器后端服务无法访问问题 (PowerShell版本)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "修复容器后端服务" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. 停止容器
Write-Host "1. 停止容器..." -ForegroundColor Yellow
docker-compose stop rqa2025-app

# 2. 重新构建镜像（如果需要）
Write-Host "2. 重新构建镜像..." -ForegroundColor Yellow
docker-compose build rqa2025-app

# 3. 启动容器
Write-Host "3. 启动容器..." -ForegroundColor Yellow
docker-compose up -d rqa2025-app

# 4. 等待服务启动
Write-Host "4. 等待服务启动（45秒）..." -ForegroundColor Yellow
Start-Sleep -Seconds 45

# 5. 检查容器状态
Write-Host "5. 检查容器状态..." -ForegroundColor Yellow
docker ps | Select-String "rqa2025-app"

# 6. 测试健康检查
Write-Host "6. 测试健康检查端点..." -ForegroundColor Yellow
docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())" 2>&1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "修复完成" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
