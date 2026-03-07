#!/bin/bash

# RQA2025 云原生部署脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
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

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装"
        exit 1
    fi
    
    # 检查docker
    if ! command -v docker &> /dev/null; then
        log_error "docker 未安装"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 构建镜像
build_images() {
    log_info "构建Docker镜像..."
    
    # 构建特征工程服务镜像
    log_info "构建特征工程服务镜像..."
    docker build -f deploy/docker/features-service.Dockerfile -t rqa2025/features-service:latest .
    
    # 构建数据服务镜像
    log_info "构建数据服务镜像..."
    docker build -f deploy/docker/data-service.Dockerfile -t rqa2025/data-service:latest .
    
    # 构建模型服务镜像
    log_info "构建模型服务镜像..."
    docker build -f deploy/docker/model-service.Dockerfile -t rqa2025/model-service:latest .
    
    log_success "镜像构建完成"
}

# 推送镜像
push_images() {
    log_info "推送Docker镜像..."
    
    # 推送特征工程服务镜像
    docker push rqa2025/features-service:latest
    
    # 推送数据服务镜像
    docker push rqa2025/data-service:latest
    
    # 推送模型服务镜像
    docker push rqa2025/model-service:latest
    
    log_success "镜像推送完成"
}

# 部署到Kubernetes
deploy_to_kubernetes() {
    local environment=$1
    
    log_info "部署到Kubernetes ($environment)..."
    
    # 创建命名空间
    kubectl apply -f deploy/kubernetes/namespace.yaml
    
    # 创建存储
    kubectl apply -f deploy/kubernetes/storage.yaml
    
    # 等待PVC创建完成
    log_info "等待持久化卷创建..."
    kubectl wait --for=condition=Bound pvc/rqa2025-data-pvc -n rqa2025 --timeout=300s
    kubectl wait --for=condition=Bound pvc/rqa2025-cache-pvc -n rqa2025 --timeout=300s
    kubectl wait --for=condition=Bound pvc/rqa2025-logs-pvc -n rqa2025 --timeout=300s
    kubectl wait --for=condition=Bound pvc/rqa2025-models-pvc -n rqa2025 --timeout=300s
    
    # 部署服务
    kubectl apply -f deploy/kubernetes/features-service.yaml
    kubectl apply -f deploy/kubernetes/data-service.yaml
    kubectl apply -f deploy/kubernetes/model-service.yaml
    
    # 等待部署完成
    log_info "等待服务部署完成..."
    kubectl rollout status deployment/features-service -n rqa2025 --timeout=300s
    kubectl rollout status deployment/data-service -n rqa2025 --timeout=300s
    kubectl rollout status deployment/model-service -n rqa2025 --timeout=300s
    
    log_success "部署完成"
}

# 验证部署
verify_deployment() {
    log_info "验证部署..."
    
    # 检查Pod状态
    kubectl get pods -n rqa2025
    
    # 检查服务状态
    kubectl get services -n rqa2025
    
    # 检查健康状态
    log_info "检查服务健康状态..."
    
    # 等待服务就绪
    sleep 30
    
    # 检查特征工程服务
    if kubectl get pods -n rqa2025 -l app=features-service --field-selector=status.phase=Running | grep -q features-service; then
        log_success "特征工程服务运行正常"
    else
        log_error "特征工程服务运行异常"
        exit 1
    fi
    
    # 检查数据服务
    if kubectl get pods -n rqa2025 -l app=data-service --field-selector=status.phase=Running | grep -q data-service; then
        log_success "数据服务运行正常"
    else
        log_error "数据服务运行异常"
        exit 1
    fi
    
    # 检查模型服务
    if kubectl get pods -n rqa2025 -l app=model-service --field-selector=status.phase=Running | grep -q model-service; then
        log_success "模型服务运行正常"
    else
        log_error "模型服务运行异常"
        exit 1
    fi
    
    log_success "部署验证完成"
}

# 清理资源
cleanup() {
    log_info "清理资源..."
    
    # 删除部署
    kubectl delete -f deploy/kubernetes/features-service.yaml --ignore-not-found=true
    kubectl delete -f deploy/kubernetes/data-service.yaml --ignore-not-found=true
    kubectl delete -f deploy/kubernetes/model-service.yaml --ignore-not-found=true
    
    # 删除存储
    kubectl delete -f deploy/kubernetes/storage.yaml --ignore-not-found=true
    
    # 删除命名空间
    kubectl delete -f deploy/kubernetes/namespace.yaml --ignore-not-found=true
    
    log_success "清理完成"
}

# 显示帮助
show_help() {
    echo "RQA2025 云原生部署脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build         构建Docker镜像"
    echo "  push          推送Docker镜像"
    echo "  deploy        部署到Kubernetes"
    echo "  verify        验证部署"
    echo "  cleanup       清理资源"
    echo "  all           执行完整部署流程"
    echo "  help          显示此帮助信息"
    echo ""
    echo "环境变量:"
    echo "  KUBECONFIG    Kubernetes配置文件路径"
    echo "  DOCKER_REGISTRY Docker镜像注册表地址"
}

# 主函数
main() {
    case "${1:-help}" in
        build)
            check_dependencies
            build_images
            ;;
        push)
            check_dependencies
            push_images
            ;;
        deploy)
            check_dependencies
            deploy_to_kubernetes "production"
            ;;
        verify)
            check_dependencies
            verify_deployment
            ;;
        cleanup)
            check_dependencies
            cleanup
            ;;
        all)
            check_dependencies
            build_images
            push_images
            deploy_to_kubernetes "production"
            verify_deployment
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"
