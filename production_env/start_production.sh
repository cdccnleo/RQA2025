#!/bin/bash
# RQA2025生产环境启动脚本

set -e

echo "🚀 启动RQA2025生产环境..."

# 检查Docker和docker-compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 启动服务
echo "🐳 启动Docker服务..."
docker-compose up -d

echo "⏳ 等待服务启动..."
sleep 60

# 运行健康检查
echo "🔍 执行健康检查..."
if ./health_check.sh; then
    echo ""
    echo "✅ 生产环境启动成功！"
    echo ""
    echo "📊 服务状态:"
    echo "  • API服务:    http://localhost:8000"
    echo "  • Grafana:    http://localhost:3000 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
    echo ""
    echo "🔧 管理命令:"
    echo "  • 查看日志:   docker-compose logs -f"
    echo "  • 停止服务:   docker-compose down"
    echo "  • 重启服务:   docker-compose restart"
else
    echo "❌ 健康检查失败，请检查服务状态"
    echo "查看日志: docker-compose logs"
    exit 1
fi