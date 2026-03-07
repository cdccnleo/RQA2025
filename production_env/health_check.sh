#!/bin/bash
# RQA2025生产环境健康检查脚本

set -e

echo "🔍 RQA2025生产环境健康检查"

# 检查服务状态
echo "📋 检查服务状态..."
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ 没有运行中的服务"
    exit 1
fi

# 检查PostgreSQL
echo "🗄️ 检查PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U rqa_user -d rqa2025_prod > /dev/null 2>&1; then
    echo "✅ PostgreSQL: 正常"
else
    echo "❌ PostgreSQL: 异常"
    exit 1
fi

# 检查Redis
echo "🔴 检查Redis..."
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis: 正常"
else
    echo "❌ Redis: 异常"
    exit 1
fi

# 检查应用
echo "🚀 检查应用服务..."
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 应用服务: 正常"
else
    echo "❌ 应用服务: 异常"
    exit 1
fi

# 检查监控 (如果存在)
if docker-compose ps | grep -q prometheus; then
    echo "📊 检查Prometheus..."
    if curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo "✅ Prometheus: 正常"
    else
        echo "❌ Prometheus: 异常"
    fi
fi

if docker-compose ps | grep -q grafana; then
    echo "📈 检查Grafana..."
    if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo "✅ Grafana: 正常"
    else
        echo "❌ Grafana: 异常"
    fi
fi

echo ""
echo "🎉 所有服务健康检查通过！"