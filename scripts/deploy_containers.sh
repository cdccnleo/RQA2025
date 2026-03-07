#!/bin/bash
# RQA2025 容器部署脚本
# RQA2025 Container Deployment Script

set -e  # 遇到错误立即退出

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

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi

    log_success "依赖检查通过"
}

# 准备环境
prepare_environment() {
    log_info "准备部署环境..."

    # 创建必要的目录
    mkdir -p logs data cache backups monitoring/prometheus monitoring/grafana/provisioning/datasources monitoring/grafana/provisioning/dashboards monitoring/loki monitoring/promtail

    # 检查环境变量文件
    if [ ! -f ".env.production" ]; then
        log_warning ".env.production 文件不存在，从模板创建..."
        if [ -f ".env.production.template" ]; then
            cp .env.production.template .env.production
            log_warning "已创建 .env.production 文件，请编辑其中的配置值"
            log_warning "编辑完成后重新运行此脚本"
            exit 1
        else
            log_error ".env.production.template 文件不存在"
            exit 1
        fi
    fi

    # 检查生产配置文件
    if [ ! -f "production_config.yml" ]; then
        log_error "production_config.yml 文件不存在"
        exit 1
    fi

    log_success "环境准备完成"
}

# 构建镜像
build_images() {
    log_info "构建应用镜像..."

    # 使用BuildKit加速构建
    export DOCKER_BUILDKIT=1

    docker-compose -f docker-compose.prod.yml build --parallel

    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."

    # 启动基础设施服务
    docker-compose -f docker-compose.prod.yml up -d postgres redis minio

    log_info "等待基础设施服务启动..."
    sleep 30

    # 检查服务健康状态
    check_service_health "postgres" "pg_isready -U rqa2025_admin -d rqa2025_prod"
    check_service_health "redis" "redis-cli ping"
    check_timescaledb_health
    check_postgres_exporter_health
    check_service_health "minio" "curl -f http://localhost:9000/minio/health/live"

    # 初始化数据库
    log_info "初始化数据库..."
    docker-compose -f docker-compose.prod.yml exec -T postgres psql -U rqa2025_admin -d rqa2025_prod -f /docker-entrypoint-initdb.d/init-db.sql || true

    # 启动监控服务
    docker-compose -f docker-compose.prod.yml up -d prometheus grafana loki promtail node-exporter cadvisor

    log_info "等待监控服务启动..."
    sleep 20

    # 启动应用服务
    docker-compose -f docker-compose.prod.yml up -d app nginx

    log_success "服务启动完成"
}

# 检查服务健康状态
check_service_health() {
    local service_name=$1
    local health_check=$2
    local max_attempts=30
    local attempt=1

    log_info "检查 $service_name 服务健康状态..."

    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f docker-compose.prod.yml exec -T $service_name $health_check &>/dev/null; then
            log_success "$service_name 服务健康检查通过"
            return 0
        fi

        log_info "等待 $service_name 服务启动... (尝试 $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done

    log_error "$service_name 服务启动失败"
    return 1
}

# 检查TimescaleDB扩展
check_timescaledb_health() {
    local max_attempts=20
    local attempt=1

    log_info "检查TimescaleDB扩展状态..."

    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f docker-compose.prod.yml exec -T postgres psql -U rqa2025_admin -d rqa2025_prod -c "SELECT 1 FROM timescaledb_information.hypertables LIMIT 1;" &>/dev/null; then
            log_success "TimescaleDB扩展检查通过"

            # 检查TimescaleDB版本信息
            docker-compose -f docker-compose.prod.yml exec -T postgres psql -U rqa2025_admin -d rqa2025_prod -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';" | grep -v "extversion\|--\|row" | xargs || true

            return 0
        fi

        log_info "等待TimescaleDB初始化... (尝试 $attempt/$max_attempts)"
        sleep 3
        ((attempt++))
    done

    log_error "TimescaleDB扩展初始化失败"
    return 1
}

# 检查PostgreSQL Exporter
check_postgres_exporter_health() {
    local max_attempts=15
    local attempt=1

    log_info "检查PostgreSQL Exporter状态..."

    while [ $attempt -le $max_attempts ]; do
        if curl -f --max-time 5 http://localhost:9187/metrics &>/dev/null; then
            log_success "PostgreSQL Exporter检查通过"
            return 0
        fi

        log_info "等待PostgreSQL Exporter启动... (尝试 $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done

    log_error "PostgreSQL Exporter启动失败"
    return 1
}

# 初始化MinIO
init_minio() {
    log_info "初始化MinIO存储桶..."

    # 等待MinIO完全启动
    sleep 10

    # 创建存储桶
    docker run --rm --network rqa2025-network \
        -e MINIO_ACCESS_KEY=minioadmin \
        -e MINIO_SECRET_KEY=minioadmin \
        minio/mc:latest \
        alias set myminio http://minio:9000 minioadmin minioadmin

    docker run --rm --network rqa2025-network \
        -e MINIO_ACCESS_KEY=minioadmin \
        -e MINIO_SECRET_KEY=minioadmin \
        minio/mc:latest \
        mb myminio/rqa2025-data

    docker run --rm --network rqa2025-network \
        -e MINIO_ACCESS_KEY=minioadmin \
        -e MINIO_SECRET_KEY=minioadmin \
        minio/mc:latest \
        mb myminio/rqa2025-backups

    docker run --rm --network rqa2025-network \
        -e MINIO_ACCESS_KEY=minioadmin \
        -e MINIO_SECRET_KEY=minioadmin \
        minio/mc:latest \
        mb myminio/rqa2025-temp

    log_success "MinIO存储桶初始化完成"
}

# 配置监控面板
setup_monitoring() {
    log_info "配置监控面板..."

    # 等待Grafana启动
    sleep 15

    # 这里可以添加自动配置Grafana数据源和仪表板的脚本
    # 暂时手动配置

    log_info "Grafana访问地址: http://localhost:3000"
    log_info "默认用户名: admin"
    log_info "默认密码: GrafanaAdmin123!"
    log_info "请手动配置数据源指向: http://prometheus:9090"

    log_success "监控配置完成"
}

# 显示部署信息
show_deployment_info() {
    echo
    log_success "🎉 RQA2025 容器部署完成！"
    echo
    echo "📊 服务访问地址:"
    echo "  🌐 Web应用:     http://localhost"
    echo "  📱 API服务:     http://localhost:8000"
    echo "  📊 API文档:     http://localhost:8000/docs"
    echo "  📈 Grafana:     http://localhost:3000 (admin/GrafanaAdmin123!)"
    echo "  📊 Prometheus:  http://localhost:9090"
    echo "  📦 MinIO:       http://localhost:9000 (minioadmin/minioadmin)"
    echo "  📝 Loki日志:    http://localhost:3100"
    echo
    echo "🔍 健康检查:"
    echo "  curl http://localhost:8000/health"
    echo
    echo "📋 管理命令:"
    echo "  # 查看服务状态"
    echo "  docker-compose -f docker-compose.prod.yml ps"
    echo
    echo "  # 查看服务日志"
    echo "  docker-compose -f docker-compose.prod.yml logs -f app"
    echo
    echo "  # 重启服务"
    echo "  docker-compose -f docker-compose.prod.yml restart app"
    echo
    echo "  # 停止所有服务"
    echo "  docker-compose -f docker-compose.prod.yml down"
    echo
    echo "⚠️  重要提醒:"
    echo "  1. 请及时修改默认密码"
    echo "  2. 配置备份策略"
    echo "  3. 设置监控告警"
    echo "  4. 配置日志轮转"
}

# 主函数
main() {
    echo "🚀 RQA2025 量化交易系统容器部署"
    echo "=================================="

    local skip_build=false
    local skip_monitoring=false

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                skip_build=true
                shift
                ;;
            --skip-monitoring)
                skip_monitoring=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                echo "使用方法: $0 [--skip-build] [--skip-monitoring]"
                exit 1
                ;;
        esac
    done

    # 执行部署步骤
    check_dependencies
    prepare_environment

    if [ "$skip_build" = false ]; then
        build_images
    else
        log_info "跳过镜像构建"
    fi

    start_services
    init_minio

    if [ "$skip_monitoring" = false ]; then
        setup_monitoring
    else
        log_info "跳过监控配置"
    fi

    show_deployment_info

    log_success "🎊 部署完成！系统已就绪"
}

# 执行主函数
main "$@"