#!/bin/bash

# 启动持续测试覆盖率监控系统
# 用法: ./start_coverage_monitor.sh

echo "🚀 启动持续测试覆盖率监控系统..."

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 创建必要的目录
mkdir -p logs
mkdir -p data
mkdir -p reports/coverage_monitoring

# 启动监控系统
python scripts/coverage_continuous_monitor.py \
    --project-root "$PROJECT_ROOT" \
    --command start

echo "✅ 监控系统启动完成"