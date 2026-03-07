#!/bin/bash

# RQA2025 核心服务部署脚本
# 使用方法: ./deploy_services.sh [environment]

set -e

# 配置变量
ENVIRONMENT=${1:-production}
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DEPLOY_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
    
    # 验证镜像
    if docker images | grep -q "rqa2025/api"; then
        log_info "API镜像构建成功"
    else
        log_error "API镜像构建失败"
        exit 1
    fi
    
    if docker images | grep -q "rqa2025/inference"; then
        log_info "推理引擎镜像构建成功"
    else
        log_error "推理引擎镜像构建失败"
        exit 1
    fi
}

# 部署蓝环境
deploy_blue_environment() {
    log_info "部署蓝环境..."
    
    cd "$DEPLOY_DIR"
    
    # 创建蓝环境配置
    cat > docker-compose.blue.yml << 'EOF'
version: '3.8'

services:
  rqa2025-api-blue:
    image: rqa2025/api:latest
    container_name: rqa2025-api-blue
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - SERVICE_COLOR=blue
      - REDIS_CLUSTER_HOSTS=192.168.1.10:6379,192.168.1.11:6379,192.168.1.12:6379
      - DATABASE_URL=postgresql://rqa2025:password@192.168.1.40:5432/rqa2025
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    ports:
      - "8000:8000"
    volumes:
      - /var/log/rqa2025:/app/logs
      - /etc/rqa2025/config:/app/config
    networks:
      - rqa2025-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  inference-engine-blue:
    image: rqa2025/inference:latest
    container_name: rqa2025-inference-blue
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - SERVICE_COLOR=blue
      - MAX_WORKERS=4
      - BATCH_SIZE=32
      - ENABLE_CACHE=true
      - CACHE_TTL=3600
      - REDIS_CLUSTER_HOSTS=192.168.1.10:6379,192.168.1.11:6379,192.168.1.12:6379
      - LOG_LEVEL=INFO
    ports:
      - "8001:8001"
    volumes:
      - /var/log/rqa2025:/app/logs
      - /etc/rqa2025/config:/app/config
      - /var/lib/rqa2025/models:/app/models
    networks:
      - rqa2025-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  rqa2025-network:
    driver: bridge
EOF

    # 启动蓝环境
    docker-compose -f docker-compose.blue.yml up -d
    
    # 等待服务启动
    log_info "等待蓝环境服务启动..."
    sleep 30
    
    # 健康检查
    check_service_health "blue"
}

# 部署绿环境
deploy_green_environment() {
    log_info "部署绿环境..."
    
    cd "$DEPLOY_DIR"
    
    # 创建绿环境配置
    cat > docker-compose.green.yml << 'EOF'
version: '3.8'

services:
  rqa2025-api-green:
    image: rqa2025/api:latest
    container_name: rqa2025-api-green
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - SERVICE_COLOR=green
      - REDIS_CLUSTER_HOSTS=192.168.1.10:6379,192.168.1.11:6379,192.168.1.12:6379
      - DATABASE_URL=postgresql://rqa2025:password@192.168.1.40:5432/rqa2025
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    ports:
      - "8002:8000"
    volumes:
      - /var/log/rqa2025:/app/logs
      - /etc/rqa2025/config:/app/config
    networks:
      - rqa2025-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  inference-engine-green:
    image: rqa2025/inference:latest
    container_name: rqa2025-inference-green
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - SERVICE_COLOR=green
      - MAX_WORKERS=4
      - BATCH_SIZE=32
      - ENABLE_CACHE=true
      - CACHE_TTL=3600
      - REDIS_CLUSTER_HOSTS=192.168.1.10:6379,192.168.1.11:6379,192.168.1.12:6379
      - LOG_LEVEL=INFO
    ports:
      - "8003:8001"
    volumes:
      - /var/log/rqa2025:/app/logs
      - /etc/rqa2025/config:/app/config
      - /var/lib/rqa2025/models:/app/models
    networks:
      - rqa2025-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  rqa2025-network:
    driver: bridge
EOF

    # 启动绿环境
    docker-compose -f docker-compose.green.yml up -d
    
    # 等待服务启动
    log_info "等待绿环境服务启动..."
    sleep 30
    
    # 健康检查
    check_service_health "green"
}

# 检查服务健康状态
check_service_health() {
    local environment=$1
    log_info "检查${environment}环境服务健康状态..."
    
    local max_retries=10
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        # 检查API服务
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_info "${environment}环境API服务健康检查通过"
        else
            log_warn "${environment}环境API服务健康检查失败"
        fi
        
        # 检查推理引擎
        if curl -f http://localhost:8001/health &> /dev/null; then
            log_info "${environment}环境推理引擎健康检查通过"
        else
            log_warn "${environment}环境推理引擎健康检查失败"
        fi
        
        # 如果两个服务都健康，退出循环
        if curl -f http://localhost:8000/health &> /dev/null && \
           curl -f http://localhost:8001/health &> /dev/null; then
            log_info "${environment}环境所有服务健康检查通过"
            return 0
        fi
        
        log_warn "服务健康检查失败，重试中... ($((retry_count + 1))/$max_retries)"
        retry_count=$((retry_count + 1))
        sleep 10
    done
    
    log_error "${environment}环境服务健康检查失败"
    return 1
}

# 配置负载均衡
configure_load_balancer() {
    log_info "配置负载均衡..."
    
    cd "$DEPLOY_DIR"
    
    # 创建Nginx配置目录
    mkdir -p nginx/sites-available
    mkdir -p nginx/sites-enabled
    
    # 生成Nginx配置
    cat > nginx/sites-available/rqa2025 << 'EOF'
upstream rqa2025_backend {
    least_conn;
    server rqa2025-api-blue:8000 max_fails=3 fail_timeout=30s;
    server rqa2025-api-green:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream inference_backend {
    least_conn;
    server rqa2025-inference-blue:8001 max_fails=3 fail_timeout=30s;
    server rqa2025-inference-green:8001 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name api.rqa2025.com;
    
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
        
        # 超时设置
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
        
        # 推理服务超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # 监控端点
    location /metrics {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

    # 启动Nginx
    docker-compose -f docker-compose.yml up -d nginx
    
    # 验证负载均衡
    sleep 10
    if curl -f http://localhost/health &> /dev/null; then
        log_info "负载均衡配置成功"
    else
        log_error "负载均衡配置失败"
        exit 1
    fi
}

# 部署监控服务
deploy_monitoring_services() {
    log_info "部署监控服务..."
    
    cd "$DEPLOY_DIR"
    
    # 启动监控服务
    docker-compose -f docker-compose.yml up -d prometheus grafana alertmanager
    
    # 等待监控服务启动
    log_info "等待监控服务启动..."
    sleep 30
    
    # 验证监控服务
    if curl -f http://localhost:9090/api/v1/targets &> /dev/null; then
        log_info "Prometheus启动成功"
    else
        log_warn "Prometheus启动异常"
    fi
    
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log_info "Grafana启动成功"
    else
        log_warn "Grafana启动异常"
    fi
}

# 部署日志服务
deploy_logging_services() {
    log_info "部署日志服务..."
    
    cd "$DEPLOY_DIR"
    
    # 启动日志服务
    docker-compose -f docker-compose.yml up -d elasticsearch logstash kibana
    
    # 等待日志服务启动
    log_info "等待日志服务启动..."
    sleep 60
    
    # 验证日志服务
    if curl -f http://localhost:9200/_cluster/health &> /dev/null; then
        log_info "Elasticsearch启动成功"
    else
        log_warn "Elasticsearch启动异常"
    fi
}

# 主函数
main() {
    log_info "开始部署RQA2025核心服务..."
    
    build_images
    deploy_blue_environment
    deploy_green_environment
    configure_load_balancer
    deploy_monitoring_services
    deploy_logging_services
    
    log_info "核心服务部署完成！"
    log_info "蓝环境: http://localhost:8000"
    log_info "绿环境: http://localhost:8002"
    log_info "负载均衡: http://localhost"
    log_info "监控面板: http://localhost:3000"
}

# 执行主函数
main "$@" 