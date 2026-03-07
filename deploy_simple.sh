#!/bin/bash

# RQA2025 简化部署脚本
echo "🚀 部署RQA2025量化交易系统（简化版）..."

# 构建镜像
echo "🔨 构建镜像..."
docker build -t rqa2025:latest .

# 启动后端服务
echo "🚀 启动后端API服务..."
docker run -d \
  --name rqa2025-app \
  -p 8000:8000 \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/config:/app/config" \
  -e RQA_ENV=production \
  -e PYTHONPATH=/app:/app/src:/app/scripts \
  rqa2025:latest \
  python scripts/start_api_server.py

# 等待服务启动
sleep 5

# 启动前端服务
echo "🌐 启动前端Nginx服务..."
docker run -d \
  --name rqa2025-web \
  -p 8080:80 \
  -v "$(pwd)/web-static:/usr/share/nginx/html" \
  -v "$(pwd)/nginx/nginx.conf:/etc/nginx/nginx.conf" \
  --link rqa2025-app \
  nginx:alpine

echo "✅ 部署完成！"
echo "📋 服务状态:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"

echo ""
echo "🌐 访问地址:"
echo "  前端: http://localhost:8080"
echo "  API: http://localhost:8000"
echo "  健康检查: http://localhost:8000/health"