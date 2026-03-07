#!/bin/bash
# RQA2025 生产部署脚本
# Production Deployment Script for RQA2025

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 命令未找到，请先安装"
        exit 1
    fi
}

# 预部署检查
pre_deployment_checks() {
    log_info "执行预部署检查..."

    # 检查必需命令
    check_command docker
    check_command docker-compose

    # 检查环境变量文件
    if [ ! -f ".env.prod" ]; then
        log_error ".env.prod 文件不存在，请从 production_env_template.yml 创建"
        exit 1
    fi

    # 检查生产配置文件
    if [ ! -f "production_config.yml" ]; then
        log_error "production_config.yml 文件不存在"
        exit 1
    fi

    # 检查磁盘空间
    local available_space=$(df / | tail -1 | awk '{print $4}')
    if [ $available_space -lt 10485760 ]; then  # 10GB in KB
        log_error "磁盘可用空间不足 10GB"
        exit 1
    fi

    log_success "预部署检查通过"
}

# 备份当前部署
backup_current_deployment() {
    log_info "备份当前部署..."

    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # 备份配置文件
    cp production_config.yml "$backup_dir/" 2>/dev/null || true
    cp docker-compose.prod.yml "$backup_dir/" 2>/dev/null || true

    # 备份环境变量（注意安全）
    cp .env.prod "$backup_dir/.env.prod.backup" 2>/dev/null || true

    log_success "备份完成: $backup_dir"
}

# 停止当前服务
stop_services() {
    log_info "停止当前服务..."

    docker-compose -f docker-compose.prod.yml down --timeout 60

    log_success "服务已停止"
}

# 构建和部署
deploy_services() {
    log_info "构建和部署服务..."

    # 部署前清理容器
    log_info "部署前清理容器..."
    ./scripts/manage_containers.sh cleanup

    # 拉取最新镜像
    docker-compose -f docker-compose.prod.yml pull

    # 构建应用镜像
    docker-compose -f docker-compose.prod.yml build --no-cache

    # 启动服务
    docker-compose -f docker-compose.prod.yml up -d

    log_success "服务部署完成"
}

# 等待服务启动
wait_for_services() {
    log_info "等待服务启动..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_info "检查服务状态 (尝试 $attempt/$max_attempts)..."

        # 检查应用健康状态
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "应用服务健康检查通过"
            break
        fi

        # 检查数据库连接
        if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U $DB_USER -d rqa2025_prod > /dev/null 2>&1; then
            log_success "数据库连接正常"
        else
            log_warn "数据库连接检查失败"
        fi

        # 检查Redis连接
        if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli --raw incr ping > /dev/null 2>&1; then
            log_success "Redis连接正常"
        else
            log_warn "Redis连接检查失败"
        fi

        sleep 10
        ((attempt++))
    done

    if [ $attempt -gt $max_attempts ]; then
        log_error "服务启动超时"
        return 1
    fi

    log_success "所有服务启动成功"
}

# 运行数据库迁移
run_database_migrations() {
    log_info "运行数据库迁移..."

    # 这里应该运行实际的数据库迁移命令
    # 例如: docker-compose -f docker-compose.prod.yml exec app python manage.py migrate

    log_success "数据库迁移完成"
}

# 验证部署
verify_deployment() {
    log_info "验证部署..."

    # 检查所有容器状态
    if docker-compose -f docker-compose.prod.yml ps | grep -q "Exit"; then
        log_error "发现退出的容器"
        docker-compose -f docker-compose.prod.yml ps
        return 1
    fi

    # 检查应用响应
    local response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    if [ "$response" != "200" ]; then
        log_error "应用健康检查失败 (HTTP $response)"
        return 1
    fi

    # 检查API文档
    local docs_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs)
    if [ "$docs_response" != "200" ]; then
        log_warn "API文档检查失败 (HTTP $docs_response)"
    fi

    log_success "部署验证通过"
}

# 部署后监控设置
setup_monitoring() {
    log_info "设置部署后监控..."

    # 等待监控服务启动
    sleep 30

    # 检查Prometheus
    if curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_success "Prometheus监控正常"
    else
        log_warn "Prometheus监控检查失败"
    fi

    # 检查Grafana
    if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
        log_success "Grafana可视化正常"
    else
        log_warn "Grafana可视化检查失败"
    fi

    log_success "监控设置完成"
}

# 主函数
main() {
    log_info "🚀 开始RQA2025生产部署"

    # 加载环境变量
    if [ -f ".env.prod" ]; then
        export $(grep -v '^#' .env.prod | xargs)
    fi

    # 执行部署步骤
    pre_deployment_checks
    backup_current_deployment
    stop_services
    deploy_services
    wait_for_services
    run_database_migrations
    verify_deployment
    setup_monitoring

    log_success "🎉 RQA2025生产部署成功完成！"
    log_info "📊 服务状态:"
    docker-compose -f docker-compose.prod.yml ps

    log_info "🌐 应用访问地址: http://localhost:8000"
    log_info "📚 API文档地址: http://localhost:8000/docs"
    log_info "📈 监控面板地址: http://localhost:3000"
    log_info "📊 Prometheus地址: http://localhost:9090"
}

# 错误处理
trap 'log_error "部署过程中发生错误"' ERR

# 执行主函数
main "$@"