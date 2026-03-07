#!/bin/bash

# RQA2025 Docker镜像批量拉取脚本
# Batch Pull Script for RQA2025 Docker Images

echo "🚀 RQA2025 Docker镜像批量拉取"
echo "================================="

# 定义需要拉取的镜像列表
IMAGES=(
    "python:3.9-slim"
    "postgres:15-alpine"
    "redis:7-alpine"
    "nginx:alpine"
    "prom/prometheus:latest"
    "grafana/grafana:latest"
)

echo "📦 需要拉取的镜像:"
for image in "${IMAGES[@]}"; do
    echo "   - $image"
done

echo
echo "🔄 开始拉取镜像..."
echo

total=${#IMAGES[@]}
success=0
failed=0

for i in "${!IMAGES[@]}"; do
    image="${IMAGES[$i]}"
    echo "[$((i+1))/$total] 拉取镜像: $image"

    if docker pull "$image"; then
        echo "✅ $image 拉取成功"
        ((success++))
    else
        echo "❌ $image 拉取失败"
        ((failed++))
    fi
    echo
done

echo "📊 拉取结果统计:"
echo "   总计镜像: $total"
echo "   成功: $success"
echo "   失败: $failed"

if [ $failed -eq 0 ]; then
    echo
    echo "🎉 所有镜像拉取完成！"
    echo
    echo "🏗️ 接下来可以构建应用镜像:"
    echo "   docker build -t rqa2025:latest ."
    echo
    echo "🚀 启动服务:"
    echo "   docker-compose -f docker-compose.prod.yml up -d"
else
    echo
    echo "⚠️ 部分镜像拉取失败，请检查网络连接或尝试手动拉取失败的镜像"
    echo
    echo "🔄 重试命令:"
    for image in "${IMAGES[@]}"; do
        echo "   docker pull $image"
    done
fi


