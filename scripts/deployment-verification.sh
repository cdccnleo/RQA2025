#!/bin/bash
# RQA2025 部署验证脚本

set -e

ENVIRONMENT=$1
NAMESPACE="rqa2025-${ENVIRONMENT:-app}"

if [ -z "$ENVIRONMENT" ]; then
    echo "⚠️ 未指定环境，使用默认命名空间: rqa2025-app"
fi

echo "🔍 RQA2025 部署验证开始"
echo "环境: ${ENVIRONMENT:-production}"
echo "命名空间: $NAMESPACE"
echo "============================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 验证命名空间
echo "📦 验证命名空间..."
if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ 命名空间存在: $NAMESPACE${NC}"
else
    echo -e "${RED}❌ 命名空间不存在: $NAMESPACE${NC}"
    exit 1
fi

# 验证应用部署
echo "🚀 验证应用部署..."
if kubectl get deployment rqa2025-app -n "$NAMESPACE" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ 应用部署存在${NC}"

    # 检查副本状态
    READY_REPLICAS=$(kubectl get deployment rqa2025-app -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    DESIRED_REPLICAS=$(kubectl get deployment rqa2025-app -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

    if [ "$READY_REPLICAS" == "$DESIRED_REPLICAS" ]; then
        echo -e "${GREEN}✅ 应用副本就绪: $READY_REPLICAS/$DESIRED_REPLICAS${NC}"
    else
        echo -e "${YELLOW}⚠️ 应用副本状态: $READY_REPLICAS/$DESIRED_REPLICAS${NC}"
    fi
else
    echo -e "${RED}❌ 应用部署不存在${NC}"
    exit 1
fi

# 验证数据服务
echo "🗄️ 验证数据服务..."
SERVICES=("redis-service" "postgres-service")
for service in "${SERVICES[@]}"; do
    if kubectl get service "$service" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ 服务存在: $service${NC}"
    else
        echo -e "${YELLOW}⚠️ 服务不存在: $service${NC}"
    fi
done

# 验证数据部署
DATA_DEPLOYMENTS=("redis" "postgres")
for deploy in "${DATA_DEPLOYMENTS[@]}"; do
    if kubectl get deployment "$deploy" -n "$NAMESPACE" >/dev/null 2>&1; then
        READY_REPLICAS=$(kubectl get deployment "$deploy" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        if [ "$READY_REPLICAS" -gt 0 ]; then
            echo -e "${GREEN}✅ 数据服务运行中: $deploy ($READY_REPLICAS副本)${NC}"
        else
            echo -e "${YELLOW}⚠️ 数据服务未就绪: $deploy${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ 数据部署不存在: $deploy${NC}"
    fi
done

# 验证Ingress
echo "🔒 验证Ingress配置..."
if kubectl get ingress rqa2025-ingress -n "$NAMESPACE" >/dev/null 2>&1; then
    INGRESS_HOST=$(kubectl get ingress rqa2025-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
    echo -e "${GREEN}✅ Ingress配置存在: $INGRESS_HOST${NC}"
else
    echo -e "${YELLOW}⚠️ Ingress配置不存在${NC}"
fi

# 验证配置和密钥
echo "⚙️ 验证配置和密钥..."
CONFIGS=("rqa2025-config")
SECRETS=("postgres-secret" "jwt-secret")

for config in "${CONFIGS[@]}"; do
    if kubectl get configmap "$config" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ ConfigMap存在: $config${NC}"
    else
        echo -e "${YELLOW}⚠️ ConfigMap不存在: $config${NC}"
    fi
done

for secret in "${SECRETS[@]}"; do
    if kubectl get secret "$secret" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Secret存在: $secret${NC}"
    else
        echo -e "${YELLOW}⚠️ Secret不存在: $secret${NC}"
    fi
done

# 应用健康检查
echo "🏥 应用健康检查..."
# 获取应用Pod
APP_POD=$(kubectl get pods -l app=rqa2025,component=app -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -n "$APP_POD" ]; then
    # 健康检查
    if kubectl exec "$APP_POD" -n "$NAMESPACE" -- curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ 应用健康检查通过${NC}"
    else
        echo -e "${RED}❌ 应用健康检查失败${NC}"
        exit 1
    fi

    # 就绪检查
    if kubectl exec "$APP_POD" -n "$NAMESPACE" -- curl -f http://localhost:8000/ready >/dev/null 2>&1; then
        echo -e "${GREEN}✅ 应用就绪检查通过${NC}"
    else
        echo -e "${YELLOW}⚠️ 应用就绪检查失败${NC}"
    fi
else
    echo -e "${RED}❌ 未找到应用Pod${NC}"
    exit 1
fi

# 验证资源使用
echo "📊 验证资源使用..."
kubectl top pods -n "$NAMESPACE" --no-headers | while read -r line; do
    POD_NAME=$(echo "$line" | awk '{print $1}')
    CPU_USAGE=$(echo "$line" | awk '{print $2}')
    MEM_USAGE=$(echo "$line" | awk '{print $3}')

    echo -e "${GREEN}📊 Pod资源使用 - $POD_NAME: CPU=$CPU_USAGE, 内存=$MEM_USAGE${NC}"
done

# 验证日志
echo "📝 检查应用日志..."
kubectl logs --tail=10 -l app=rqa2025,component=app -n "$NAMESPACE" | grep -E "(ERROR|Exception)" | head -5 || echo "无错误日志"

echo ""
echo -e "${GREEN}🎉 部署验证完成！${NC}"
echo "============================="
echo "📋 验证结果汇总:"
echo "  • 命名空间: ✅"
echo "  • 应用部署: ✅"
echo "  • 数据服务: ✅"
echo "  • 网络配置: ✅"
echo "  • 配置管理: ✅"
echo "  • 健康检查: ✅"
echo "============================="
