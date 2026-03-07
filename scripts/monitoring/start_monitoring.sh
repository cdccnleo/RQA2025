#!/bin/bash
# 启动监控服务
echo "启动RQA2025监控服务..."

# 启动Prometheus
docker-compose -f docker-compose.monitoring.yml up -d prometheus
echo "Prometheus已启动: http://localhost:9090"

# 启动Grafana
docker-compose -f docker-compose.monitoring.yml up -d grafana
echo "Grafana已启动: http://localhost:3000 (admin/admin123)"

# 启动AlertManager
docker-compose -f docker-compose.monitoring.yml up -d alertmanager
echo "AlertManager已启动: http://localhost:9093"

echo "监控服务启动完成！"
