#!/bin/bash
# RQA2025生产环境停止脚本

echo "🛑 停止RQA2025生产环境..."

# 优雅停止
docker-compose down

echo "✅ 生产环境已停止"