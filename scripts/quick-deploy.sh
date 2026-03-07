#!/bin/bash
# RQA2025 快速部署脚本
# 支持开发环境热重载和生产环境滚动更新

set -e

# 配置
NAMESPACE=${NAMESPACE:-"rqa2025-app"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
ROLLING_UPDATE=${ROLLING_UPDATE:-"true"}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查kubectl连接
check_kubectl() {
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "无法连接到Kubernetes集群"
        exit 1
    fi
    log_info "Kubernetes集群连接正常"
}

# 检查命名空间
check_namespace() {
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_warning "命名空间 $NAMESPACE 不存在，正在创建..."
        kubectl create namespace "$NAMESPACE"
    fi
}

# 开发环境部署
deploy_dev() {
    log_info "开始开发环境部署..."

    # 检查开发环境配置
    if [ ! -f "k8s/development/rqa2025-app-deployment-dev.yaml" ]; then
        log_error "开发环境配置文件不存在"
        exit 1
    fi

    # 应用配置
    kubectl apply -f k8s/development/rqa2025-app-deployment-dev.yaml

    # 等待部署完成
    kubectl rollout status deployment/rqa2025-app-dev -n rqa2025-dev

    log_success "开发环境部署完成"
    log_info "应用将在代码变更时自动重载"
}

# 生产环境部署
deploy_prod() {
    log_info "开始生产环境部署..."

    # 检查是否有新的镜像
    local latest_image=$(docker images rqa2025 --format "{{.Repository}}:{{.Tag}}" | head -n1)
    if [ -z "$latest_image" ]; then
        log_error "未找到本地镜像，请先构建镜像"
        log_info "运行: docker build -t rqa2025:latest ."
        exit 1
    fi

    log_info "使用镜像: $latest_image"

    if [ "$ROLLING_UPDATE" = "true" ]; then
        # 滚动更新策略
        log_info "执行滚动更新..."

        # 更新镜像
        kubectl set image deployment/rqa2025-app-rolling \
          rqa2025-app=$latest_image \
          -n "$NAMESPACE"

        # 等待滚动更新完成
        kubectl rollout status deployment/rqa2025-app-rolling -n "$NAMESPACE"

        # 检查健康状态
        check_deployment_health

    else
        # 重建部署
        kubectl apply -f k8s/production/rqa2025-app-deployment.yaml
        kubectl rollout status deployment/rqa2025-app -n "$NAMESPACE"
    fi

    log_success "生产环境部署完成"
}

# 检查部署健康状态
check_deployment_health() {
    log_info "检查部署健康状态..."

    # 等待所有Pod就绪
    local timeout=300
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" \
          -l app=rqa2025,component=app \
          -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | \
          grep -o "True" | wc -l)

        local total_pods=$(kubectl get pods -n "$NAMESPACE" \
          -l app=rqa2025,component=app \
          --no-headers | wc -l)

        if [ "$ready_pods" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
            log_success "所有Pod已就绪 ($ready_pods/$total_pods)"
            return 0
        fi

        log_info "等待Pod就绪... ($ready_pods/$total_pods)"
        sleep 10
        elapsed=$((elapsed + 10))
    done

    log_error "部署健康检查超时"
    kubectl get pods -n "$NAMESPACE" -l app=rqa2025,component=app
    exit 1
}

# 前端配置更新
update_frontend() {
    log_info "更新前端配置..."

    # 更新ConfigMap
    kubectl create configmap web-static-files \
      --from-file=web-static/ \
      --dry-run=client -o yaml | \
      kubectl apply -f -

    kubectl create configmap frontend-nginx-config \
      --from-file=nginx.conf=web-static/nginx.conf \
      --dry-run=client -o yaml | \
      kubectl apply -f -

    # 触发前端Pod重启
    kubectl rollout restart deployment/rqa2025-frontend -n "$NAMESPACE"

    log_success "前端配置更新完成"
}

# 主函数
main() {
    echo "🚀 RQA2025 快速部署脚本"
    echo "环境: $ENVIRONMENT"
    echo "命名空间: $NAMESPACE"
    echo "滚动更新: $ROLLING_UPDATE"
    echo

    check_kubectl
    check_namespace

    case "$ENVIRONMENT" in
        "development"|"dev")
            deploy_dev
            ;;
        "production"|"prod")
            deploy_prod
            ;;
        "frontend"|"fe")
            update_frontend
            ;;
        *)
            log_error "无效的环境: $ENVIRONMENT"
            log_info "使用方法:"
            log_info "  $0 -e development    # 开发环境部署"
            log_info "  $0 -e production     # 生产环境部署"
            log_info "  $0 -e frontend       # 前端配置更新"
            exit 1
            ;;
    esac

    log_success "部署完成 🎉"
}

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --no-rolling)
            ROLLING_UPDATE="false"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  -e, --environment ENV    部署环境 (development/production/frontend)"
            echo "  -n, --namespace NS        Kubernetes命名空间"
            echo "  --no-rolling             禁用滚动更新"
            echo "  -h, --help               显示帮助信息"
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            exit 1
            ;;
    esac
done

main