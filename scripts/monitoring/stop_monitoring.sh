#!/bin/bash
# 停止监控服务
echo "停止RQA2025监控服务..."

docker-compose -f docker-compose.monitoring.yml down

echo "监控服务已停止！"
