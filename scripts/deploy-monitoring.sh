#!/bin/bash
# RQA2025 监控系统部署脚本

set -e

echo "📊 RQA2025 监控系统部署开始"
echo "============================="

# 检查kubectl连接
echo "🔍 检查Kubernetes集群连接..."
kubectl cluster-info >/dev/null
echo "✅ Kubernetes集群连接正常"

# 创建监控命名空间
echo "📦 创建监控命名空间..."
kubectl apply -f monitoring/prometheus-deployment.yaml
echo "✅ 命名空间和RBAC配置完成"

# 部署Prometheus
echo "🔥 部署Prometheus..."
kubectl apply -f monitoring/prometheus-deployment.yaml
echo "⏳ 等待Prometheus启动..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n monitoring
echo "✅ Prometheus部署完成"

# 部署Alertmanager
echo "🚨 部署Alertmanager..."
kubectl apply -f monitoring/alertmanager-deployment.yaml
kubectl wait --for=condition=available --timeout=300s deployment/alertmanager -n monitoring
echo "✅ Alertmanager部署完成"

# 部署Grafana
echo "📈 部署Grafana..."
kubectl apply -f monitoring/grafana-deployment.yaml
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n monitoring
echo "✅ Grafana部署完成"

# 配置Prometheus规则
echo "📋 配置Prometheus告警规则..."
kubectl apply -f monitoring/prometheus/rules.yml
echo "✅ 告警规则配置完成"

# 配置应用监控
echo "🔧 配置应用监控..."
# 为RQA2025应用添加监控注解
kubectl patch deployment rqa2025-app -n rqa2025-app -p '{
  "spec": {
    "template": {
      "metadata": {
        "annotations": {
          "prometheus.io/scrape": "true",
          "prometheus.io/port": "8000",
          "prometheus.io/path": "/metrics"
        }
      }
    }
  }
}'
echo "✅ 应用监控配置完成"

# 创建监控Ingress
echo "🔒 创建监控Ingress..."
cat > monitoring/ingress.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: monitoring
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: monitoring-auth
    nginx.ingress.kubernetes.io/auth-realm: 'Authentication Required'
spec:
  rules:
  - host: prometheus.rqa2025.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
  - host: grafana.rqa2025.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
  - host: alertmanager.rqa2025.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: alertmanager
            port:
              number: 9093
EOF

kubectl apply -f monitoring/ingress.yaml
echo "✅ 监控Ingress配置完成"

# 创建基础认证Secret
echo "🔐 创建监控访问认证..."
kubectl create secret generic monitoring-auth -n monitoring \
  --from-literal=auth=$(htpasswd -nb admin rqa2025admin | base64 -w 0) \
  --dry-run=client -o yaml | kubectl apply -f -
echo "✅ 监控认证配置完成"

# 验证部署
echo "✅ 验证监控系统部署..."
echo "Prometheus服务状态:"
kubectl get pods -l app=prometheus -n monitoring
echo ""
echo "Grafana服务状态:"
kubectl get pods -l app=grafana -n monitoring
echo ""
echo "Alertmanager服务状态:"
kubectl get pods -l app=alertmanager -n monitoring
echo ""
echo "监控服务访问地址:"
echo "  📊 Prometheus:  http://prometheus.rqa2025.example.com"
echo "  📈 Grafana:     http://grafana.rqa2025.example.com"
echo "  🚨 Alertmanager: http://alertmanager.rqa2025.example.com"
echo ""
echo "默认登录凭据:"
echo "  用户名: admin"
echo "  密码: rqa2025admin"

# 检查告警状态
echo ""
echo "🚨 检查告警状态..."
kubectl get prometheusrules -n monitoring
kubectl get alertmanagers -n monitoring

echo ""
echo "🎉 RQA2025 监控系统部署完成！"
echo "============================="
echo "📊 监控功能概览:"
echo "  • 应用性能监控 (APM)"
echo "  • 基础设施监控"
echo "  • 智能告警系统"
echo "  • 可视化仪表板"
echo "  • 告警路由和抑制"
echo ""
echo "📈 仪表板功能:"
echo "  • 服务状态总览"
echo "  • CPU/内存使用率"
echo "  • 请求率和错误率"
echo "  • 响应时间分析"
echo "  • 业务指标监控"
echo ""
echo "🚨 告警类型:"
echo "  • 应用服务告警"
echo "  • 基础设施告警"
echo "  • 业务指标告警"
echo "  • Kubernetes告警"
echo ""
echo "📧 通知渠道:"
echo "  • 邮件通知"
echo "  • Slack集成"
echo "  • PagerDuty集成"
echo ""
echo "🔧 后续配置:"
echo "  1. 更新SMTP和Slack配置"
echo "  2. 配置业务指标监控"
echo "  3. 设置告警升级策略"
echo "  4. 定制仪表板视图"




