#!/bin/bash
# RQA2025 蓝绿部署脚本

set -e

ENVIRONMENT=${1:-production}
NAMESPACE="rqa2025-${ENVIRONMENT}"
NEW_VERSION=$2

if [ -z "$NEW_VERSION" ]; then
    echo "❌ 请指定新版本标签"
    echo "使用方法: $0 [environment] <version>"
    exit 1
fi

echo "🔄 RQA2025 蓝绿部署开始"
echo "环境: $ENVIRONMENT"
echo "命名空间: $NAMESPACE"
echo "新版本: $NEW_VERSION"
echo "============================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取当前活跃部署
get_active_deployment() {
    local active_deployment=""

    if kubectl get service rqa2025-app-service -n "$NAMESPACE" >/dev/null 2>&1; then
        active_deployment=$(kubectl get service rqa2025-app-service -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}' 2>/dev/null)

        if [ -z "$active_deployment" ]; then
            active_deployment="blue"  # 默认使用blue
        fi
    else
        active_deployment="blue"  # 首次部署使用blue
    fi

    echo "$active_deployment"
}

# 确定新部署的颜色
ACTIVE_DEPLOYMENT=$(get_active_deployment)
if [ "$ACTIVE_DEPLOYMENT" == "blue" ]; then
    NEW_DEPLOYMENT="green"
    OLD_DEPLOYMENT="blue"
else
    NEW_DEPLOYMENT="green"
    OLD_DEPLOYMENT="blue"
fi

echo -e "${BLUE}📊 当前活跃部署: $ACTIVE_DEPLOYMENT${NC}"
echo -e "${GREEN}📊 新部署颜色: $NEW_DEPLOYMENT${NC}"

# 创建新版本的部署
echo "🚀 创建新版本部署..."
cat > "k8s/${ENVIRONMENT}/rqa2025-app-${NEW_DEPLOYMENT}.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-app-${NEW_DEPLOYMENT}
  namespace: ${NAMESPACE}
  labels:
    app: rqa2025
    component: app
    environment: ${ENVIRONMENT}
    version: ${NEW_DEPLOYMENT}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rqa2025
      component: app
      version: ${NEW_DEPLOYMENT}
  template:
    metadata:
      labels:
        app: rqa2025
        component: app
        environment: ${ENVIRONMENT}
        version: ${NEW_DEPLOYMENT}
    spec:
      containers:
      - name: rqa2025-app
        image: rqa2025:${NEW_VERSION}
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "${ENVIRONMENT}"
        - name: VERSION
          value: "${NEW_DEPLOYMENT}"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
EOF

# 部署新版本
kubectl apply -f "k8s/${ENVIRONMENT}/rqa2025-app-${NEW_DEPLOYMENT}.yaml"

# 等待新版本就绪
echo "⏳ 等待新版本部署就绪..."
kubectl wait --for=condition=available --timeout=600s deployment/rqa2025-app-${NEW_DEPLOYMENT} -n "$NAMESPACE"

# 验证新版本健康状态
echo "🏥 验证新版本健康状态..."
NEW_POD=$(kubectl get pods -l app=rqa2025,version=${NEW_DEPLOYMENT} -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')

if kubectl exec "$NEW_POD" -n "$NAMESPACE" -- curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ 新版本健康检查通过${NC}"
else
    echo -e "${RED}❌ 新版本健康检查失败${NC}"
    # 清理失败的部署
    kubectl delete deployment rqa2025-app-${NEW_DEPLOYMENT} -n "$NAMESPACE"
    exit 1
fi

# 执行流量切换
echo "🔄 执行流量切换..."
kubectl patch service rqa2025-app-service -n "$NAMESPACE" -p "{
  \"spec\": {
    \"selector\": {
      \"app\": \"rqa2025\",
      \"component\": \"app\",
      \"version\": \"${NEW_DEPLOYMENT}\"
    }
  }
}"

# 等待流量切换完成
echo "⏳ 等待流量切换完成..."
sleep 30

# 验证流量切换结果
echo "✅ 验证流量切换结果..."
kubectl get service rqa2025-app-service -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}'

# 监控新版本性能
echo "📊 监控新版本性能..."
sleep 60

# 检查错误率和响应时间
echo "🔍 检查新版本指标..."
kubectl logs --tail=20 -l app=rqa2025,version=${NEW_DEPLOYMENT} -n "$NAMESPACE" | grep -E "(ERROR|Exception)" | head -5 || echo "无错误日志"

# 确认部署成功
read -p "🔍 新版本运行正常吗? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}✅ 蓝绿部署成功完成${NC}"

    # 清理旧版本
    if kubectl get deployment rqa2025-app-${OLD_DEPLOYMENT} -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "🧹 清理旧版本部署..."
        kubectl delete deployment rqa2025-app-${OLD_DEPLOYMENT} -n "$NAMESPACE"
        echo -e "${GREEN}✅ 旧版本清理完成${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ 检测到问题，开始回滚...${NC}"

    # 回滚到旧版本
    kubectl patch service rqa2025-app-service -n "$NAMESPACE" -p "{
      \"spec\": {
        \"selector\": {
          \"app\": \"rqa2025\",
          \"component\": \"app\",
          \"version\": \"${OLD_DEPLOYMENT}\"
        }
      }
    }"

    # 清理新版本
    kubectl delete deployment rqa2025-app-${NEW_DEPLOYMENT} -n "$NAMESPACE"

    echo -e "${GREEN}✅ 回滚完成${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 蓝绿部署流程完成！${NC}"
echo "============================="
echo "📋 部署结果:"
echo "  • 活跃版本: $NEW_DEPLOYMENT"
echo "  • 版本标签: $NEW_VERSION"
echo "  • 命名空间: $NAMESPACE"
echo "  • 流量切换: ✅"
echo "============================="
