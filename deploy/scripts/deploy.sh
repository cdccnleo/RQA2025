#!/bin/bash

# RQA2025 生产环境自动化部署脚本
# 使用方法: ./deploy.sh [environment]

set -e

# 配置变量
ENVIRONMENT=${1:-production}
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DEPLOY_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查部署依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装"
        exit 1
    fi
    
    # 检查curl
    if ! command -v curl &> /dev/null; then
        log_error "curl未安装"
        exit 1
    fi
    
    log_info "依赖检查完成"
}

# 备份现有配置
backup_config() {
    log_info "备份现有配置..."
    
    BACKUP_DIR="/etc/rqa2025.backup.$(date +%Y%m%d_%H%M%S)"
    sudo mkdir -p "$BACKUP_DIR"
    
    if [ -d "/etc/rqa2025" ]; then
        sudo cp -r /etc/rqa2025/* "$BACKUP_DIR/"
        log_info "配置已备份到: $BACKUP_DIR"
    else
        log_warn "未找到现有配置目录"
    fi
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    sudo mkdir -p /etc/rqa2025/config
    sudo mkdir -p /var/log/rqa2025
    sudo mkdir -p /var/lib/rqa2025/models
    sudo mkdir -p /var/lib/rqa2025/cache
    
    # 设置权限
    sudo chown -R $USER:$USER /var/log/rqa2025
    sudo chown -R $USER:$USER /var/lib/rqa2025
    
    log_info "目录创建完成"
}

# 部署Redis集群
deploy_redis_cluster() {
    log_info "部署Redis集群..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        # 生产环境使用外部Redis集群
        log_info "使用外部Redis集群配置"
        cp "$DEPLOY_DIR/../config/redis_cluster.yaml" /etc/rqa2025/config/
    else
        # 开发环境启动Redis容器
        log_info "启动Redis容器..."
        docker-compose -f "$DEPLOY_DIR/../docker-compose.yml" up -d redis-proxy
    fi
    
    log_info "Redis集群部署完成"
}

# 构建Docker镜像
build_images() {
    log_info "构建Docker镜像..."
    
    cd "$PROJECT_ROOT"
    
    # 构建API镜像
    log_info "构建RQA2025 API镜像..."
    docker build -f deploy/Dockerfile -t rqa2025/api:latest .
    
    # 构建推理引擎镜像
    log_info "构建推理引擎镜像..."
    docker build -f deploy/Dockerfile.inference -t rqa2025/inference:latest .
    
    log_info "Docker镜像构建完成"
}

# 部署应用服务
deploy_services() {
    log_info "部署应用服务..."
    
    cd "$DEPLOY_DIR"
    
    # 启动服务
    docker-compose -f docker-compose.yml up -d
    
    # 等待服务启动
    log_info "等待服务启动..."
    sleep 30
    
    # 检查服务状态
    check_services_health
    
    log_info "应用服务部署完成"
}

# 检查服务健康状态
check_services_health() {
    log_info "检查服务健康状态..."
    
    local max_retries=10
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -f http://localhost/health &> /dev/null; then
            log_info "服务健康检查通过"
            return 0
        fi
        
        log_warn "服务健康检查失败，重试中... ($((retry_count + 1))/$max_retries)"
        retry_count=$((retry_count + 1))
        sleep 10
    done
    
    log_error "服务健康检查失败"
    return 1
}

# 配置监控系统
setup_monitoring() {
    log_info "配置监控系统..."
    
    # 创建监控配置目录
    mkdir -p "$DEPLOY_DIR/monitoring"
    
    # 复制监控配置文件
    if [ ! -f "$DEPLOY_DIR/monitoring/prometheus.yml" ]; then
        log_error "Prometheus配置文件不存在"
        exit 1
    fi
    
    if [ ! -f "$DEPLOY_DIR/monitoring/alert_rules.yml" ]; then
        log_error "告警规则配置文件不存在"
        exit 1
    fi
    
    # 启动监控服务
    docker-compose -f docker-compose.yml up -d prometheus grafana alertmanager
    
    log_info "监控系统配置完成"
}

# 配置负载均衡
setup_load_balancer() {
    log_info "配置负载均衡..."
    
    # 创建Nginx配置目录
    mkdir -p "$DEPLOY_DIR/nginx/sites-available"
    mkdir -p "$DEPLOY_DIR/nginx/sites-enabled"
    
    # 生成Nginx配置
    cat > "$DEPLOY_DIR/nginx/sites-available/rqa2025" << 'EOF'
upstream rqa2025_backend {
    least_conn;
    server rqa2025-api:8000 max_fails=3 fail_timeout=30s;
    server rqa2025-api-2:8000 max_fails=3 fail_timeout=30s;
    server rqa2025-api-3:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream inference_backend {
    least_conn;
    server inference-engine:8001 max_fails=3 fail_timeout=30s;
    server inference-engine-2:8001 max_fails=3 fail_timeout=30s;
    server inference-engine-3:8001 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name localhost;
    
    # 健康检查
    location /health {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # API路由
    location /api/ {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # 推理服务路由
    location /inference/ {
        proxy_pass http://inference_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # 监控端点
    location /metrics {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
    }
}
EOF
    
    # 启用配置
    ln -sf "$DEPLOY_DIR/nginx/sites-available/rqa2025" "$DEPLOY_DIR/nginx/sites-enabled/"
    
    log_info "负载均衡配置完成"
}

# 性能测试
run_performance_test() {
    log_info "运行性能测试..."
    
    # 简单的性能测试
    local test_url="http://localhost/api/health"
    local response_time=$(curl -o /dev/null -s -w "%{time_total}" "$test_url")
    
    log_info "API响应时间: ${response_time}s"
    
    if (( $(echo "$response_time < 1.0" | bc -l) )); then
        log_info "性能测试通过"
    else
        log_warn "性能测试警告: 响应时间较慢"
    fi
}

# 部署后检查
post_deployment_check() {
    log_info "执行部署后检查..."
    
    # 检查服务状态
    docker-compose -f docker-compose.yml ps
    
    # 检查端口监听
    netstat -tlnp | grep -E ':(80|8000|8001|9090|3000)' || true
    
    # 检查日志
    docker-compose -f docker-compose.yml logs --tail=20
    
    log_info "部署后检查完成"
}

# 显示部署信息
show_deployment_info() {
    log_info "=== 部署完成 ==="
    echo
    echo "服务访问地址:"
    echo "  - API服务: http://localhost/api"
    echo "  - 推理服务: http://localhost/inference"
    echo "  - 健康检查: http://localhost/health"
    echo
    echo "监控系统:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin123)"
    echo "  - AlertManager: http://localhost:9093"
    echo
    echo "日志查看:"
    echo "  - 应用日志: docker-compose logs rqa2025-api"
    echo "  - 推理日志: docker-compose logs inference-engine"
    echo
    echo "常用命令:"
    echo "  - 停止服务: docker-compose down"
    echo "  - 重启服务: docker-compose restart"
    echo "  - 查看状态: docker-compose ps"
    echo
}

# 主函数
main() {
    log_info "开始RQA2025生产环境部署..."
    log_info "部署环境: $ENVIRONMENT"
    
    check_dependencies
    backup_config
    create_directories
    deploy_redis_cluster
    build_images
    setup_load_balancer
    deploy_services
    setup_monitoring
    run_performance_test
    post_deployment_check
    show_deployment_info
    
    log_info "部署完成！"
}

# 错误处理
trap 'log_error "部署过程中发生错误，请检查日志"; exit 1' ERR

# 执行主函数
main "$@" 