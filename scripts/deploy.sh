#!/bin/bash

# RQA2025 部署脚本
# 支持多环境部署：development, staging, production

set -e

# 配置
APP_NAME="rqa2025"
DOCKER_IMAGE="${APP_NAME}:latest"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# 颜色输出
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

# 检查Docker是否运行
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    log_success "Docker is running"
}

# 检查Docker Compose
check_docker_compose() {
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_success "Docker Compose is available"
}

# 构建Docker镜像
build_image() {
    log_info "Building Docker image: $DOCKER_IMAGE"
    docker build -t "$DOCKER_IMAGE" .
    log_success "Docker image built successfully"
}

# 运行预部署检查
pre_deploy_checks() {
    log_info "Running pre-deployment checks..."

    # 检查必要的文件
    required_files=("docker-compose.yml" "Dockerfile" "requirements.txt")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done

    # 检查环境变量
    if [ -z "$ENVIRONMENT" ]; then
        ENVIRONMENT="development"
        log_warning "ENVIRONMENT not set, using default: $ENVIRONMENT"
    fi

    log_success "Pre-deployment checks passed"
}

# 部署应用
deploy_app() {
    local env=$1
    log_info "Deploying to $env environment..."

    # 设置环境变量
    export ENVIRONMENT=$env

    # 停止现有服务
    log_info "Stopping existing services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down || true

    # 启动服务
    log_info "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d

    # 等待服务启动
    log_info "Waiting for services to be ready..."
    sleep 30

    # 检查服务健康状态
    check_services_health

    log_success "Deployment to $env completed successfully"
}

# 检查服务健康状态
check_services_health() {
    log_info "Checking service health..."

    # 检查主要服务
    services=("web" "api" "worker")

    for service in "${services[@]}"; do
        if docker-compose ps | grep -q "$service.*Up"; then
            log_success "Service $service is healthy"
        else
            log_warning "Service $service might not be healthy"
        fi
    done
}

# 运行数据库迁移
run_migrations() {
    log_info "Running database migrations..."
    docker-compose exec api python manage.py migrate || log_warning "Migration failed, but continuing..."
}

# 运行健康检查
run_health_checks() {
    log_info "Running health checks..."

    # 等待服务完全启动
    sleep 10

    # 检查HTTP端点
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_warning "API health check failed"
    fi
}

# 清理函数
cleanup() {
    log_info "Cleaning up..."
    # 清理临时文件、日志等
    docker system prune -f >/dev/null 2>&1 || true
    log_success "Cleanup completed"
}

# 显示使用帮助
show_help() {
    cat << EOF
RQA2025 部署脚本

使用方法:
  $0 [选项] [环境]

环境:
  development    开发环境 (默认)
  staging       预发布环境
  production    生产环境

选项:
  -h, --help     显示此帮助信息
  --build-only   仅构建镜像，不部署
  --no-cache     构建时不使用缓存
  --verbose      详细输出

示例:
  $0 production              # 部署到生产环境
  $0 --build-only           # 仅构建镜像
  $0 --no-cache staging     # 无缓存构建并部署到预发布环境

EOF
}

# 主函数
main() {
    local build_only=false
    local no_cache=false
    local verbose=false
    local environment="development"

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --build-only)
                build_only=true
                shift
                ;;
            --no-cache)
                no_cache=true
                shift
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            development|staging|production)
                environment=$1
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 设置详细输出
    if [ "$verbose" = true ]; then
        set -x
    fi

    log_info "Starting RQA2025 deployment to $environment environment"

    # 执行部署步骤
    check_docker
    check_docker_compose
    pre_deploy_checks

    if [ "$no_cache" = true ]; then
        DOCKER_BUILDKIT=1 docker build --no-cache -t "$DOCKER_IMAGE" .
    else
        build_image
    fi

    if [ "$build_only" = true ]; then
        log_success "Build completed successfully (build-only mode)"
        exit 0
    fi

    deploy_app "$environment"
    run_migrations
    run_health_checks
    cleanup

    log_success "🎉 RQA2025 deployment completed successfully!"
    log_info "Application is available at: http://localhost:8000"
}

# 执行主函数
main "$@"