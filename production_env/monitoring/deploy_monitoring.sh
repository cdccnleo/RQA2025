#!/bin/bash
# RQA2025监控系统部署脚本

set -e

echo "🚀 部署RQA2025监控系统..."

# 检查Docker和docker-compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 创建必要的目录
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

# 生成Grafana数据源配置
cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# 生成Grafana仪表板配置
cat > monitoring/grafana/provisioning/dashboards/rqa2025.yml << EOF
apiVersion: 1
providers:
  - name: 'RQA2025'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

echo "📊 启动监控服务..."
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

echo "⏳ 等待服务启动..."
sleep 30

echo "✅ 监控系统部署完成！"
echo ""
echo "📈 访问地址:"
echo "  • Grafana:    http://localhost:3000 (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • AlertManager: http://localhost:9093"
echo ""
echo "🔍 查看服务状态:"
echo "  docker-compose -f monitoring/docker-compose.monitoring.yml ps"
echo ""
echo "🛑 停止服务:"
echo "  docker-compose -f monitoring/docker-compose.monitoring.yml down"
