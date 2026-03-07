#!/bin/bash
# 修复容器后端服务无法访问问题

echo "=========================================="
echo "修复容器后端服务"
echo "=========================================="

# 1. 停止容器
echo "1. 停止容器..."
docker-compose stop rqa2025-app

# 2. 重新构建镜像（如果需要）
echo "2. 重新构建镜像..."
docker-compose build rqa2025-app

# 3. 启动容器
echo "3. 启动容器..."
docker-compose up -d rqa2025-app

# 4. 等待服务启动
echo "4. 等待服务启动（45秒）..."
sleep 45

# 5. 检查容器状态
echo "5. 检查容器状态..."
docker ps | grep rqa2025-app

# 6. 测试健康检查
echo "6. 测试健康检查端点..."
docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())" 2>&1

echo "=========================================="
echo "修复完成"
echo "=========================================="
