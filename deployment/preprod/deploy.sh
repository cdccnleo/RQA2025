#!/bin/bash

# RQA2025 预投产环境部署脚本
# 使用方法: ./deploy.sh [up|down|restart|status|logs]

set -e

# 配置变量
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="rqa2025-preprod"
ENV_FILE=".env"

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

    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi

    # 检查curl
    if ! command -v curl &> /dev/null; then
        log_error "curl 未安装，请先安装 curl"
        exit 1
    fi

    log_success "系统依赖检查通过"
}

# 创建环境文件
create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_info "创建环境配置文件..."

        cat > "$ENV_FILE" << EOF
# RQA2025 预投产环境配置

# PostgreSQL 配置
POSTGRES_PASSWORD=rqa2025_secure_pass_preprod

# InfluxDB 配置
INFLUXDB_PASSWORD=rqa2025_influx_pass_preprod
INFLUXDB_TOKEN=rqa2025_token_preprod_12345

# Redis 配置
REDIS_PASSWORD=rqa2025_redis_pass_preprod

# Grafana 配置
GRAFANA_USER=admin
GRAFANA_PASSWORD=rqa2025_grafana_pass_preprod

# Elasticsearch 配置
ELASTIC_PASSWORD=rqa2025_elastic_pass_preprod

# 应用配置
ENV=preprod
DEBUG=false
LOG_LEVEL=INFO
EOF

        log_success "环境配置文件创建完成: $ENV_FILE"
    else
        log_info "环境配置文件已存在: $ENV_FILE"
    fi
}

# 启动服务
start_services() {
    log_info "启动预投产环境服务..."

    # 使用docker-compose或docker compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    # 启动服务
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d

    log_success "服务启动命令已执行，等待服务就绪..."
}

# 等待服务健康检查
wait_for_services() {
    log_info "等待服务启动完成..."

    services=("postgres" "influxdb" "redis" "prometheus" "grafana" "elasticsearch" "kibana" "health-monitor")
    max_attempts=60  # 最多等待10分钟
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        healthy_count=0

        for service in "${services[@]}"; do
            if $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps "$service" | grep -q "healthy\|running"; then
                healthy_count=$((healthy_count + 1))
            fi
        done

        if [ $healthy_count -eq ${#services[@]} ]; then
            log_success "所有服务已就绪! ($healthy_count/${#services[@]} 服务健康)"
            return 0
        fi

        log_info "等待服务就绪... ($healthy_count/${#services[@]} 服务就绪，尝试 $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done

    log_error "服务启动超时，请检查日志"
    return 1
}

# 验证部署
validate_deployment() {
    log_info "验证部署状态..."

    # 检查服务状态
    log_info "检查服务状态:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps

    # 验证关键端点
    endpoints=(
        "http://localhost:8000/health:Health Monitor"
        "http://localhost:9090/-/healthy:Prometheus"
        "http://localhost:3000/api/health:Grafana"
        "http://localhost:9200/_cluster/health:Elasticsearch"
        "http://localhost:5601/api/status:Kibana"
    )

    log_info "验证关键端点:"
    for endpoint in "${endpoints[@]}"; do
        url=$(echo "$endpoint" | cut -d: -f1)
        name=$(echo "$endpoint" | cut -d: -f2)

        if curl -f -s "$url" > /dev/null 2>&1; then
            log_success "✅ $name: $url"
        else
            log_warning "⚠️  $name: $url (不可访问)"
        fi
    done

    # 验证数据库连接
    log_info "验证数据库连接..."
    if $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" exec -T postgres pg_isready -U rqa2025_user -d rqa2025 > /dev/null 2>&1; then
        log_success "✅ PostgreSQL 连接正常"
    else
        log_error "❌ PostgreSQL 连接失败"
    fi

    # 验证Redis连接
    if $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" exec -T redis redis-cli -a rqa2025_redis_pass_preprod ping | grep -q "PONG"; then
        log_success "✅ Redis 连接正常"
    else
        log_error "❌ Redis 连接失败"
    fi
}

# 显示服务信息
show_service_info() {
    log_info "预投产环境服务信息:"
    echo ""
    echo "🌐 Web 界面:"
    echo "  Health Monitor: http://localhost:8000"
    echo "  Grafana:        http://localhost:3000 (admin / rqa2025_grafana_pass_preprod)"
    echo "  Kibana:         http://localhost:5601"
    echo "  Prometheus:     http://localhost:9090"
    echo ""
    echo "💾 数据库服务:"
    echo "  PostgreSQL:     localhost:5432 (rqa2025_user / rqa2025_secure_pass_preprod)"
    echo "  InfluxDB:       localhost:8086 (rqa2025_admin / rqa2025_influx_pass_preprod)"
    echo "  Redis:          localhost:6379 (password: rqa2025_redis_pass_preprod)"
    echo "  Elasticsearch:  localhost:9200"
    echo ""
    echo "📊 监控指标:"
    echo "  Health Metrics: http://localhost:8000/metrics"
    echo "  Prometheus API: http://localhost:9090/api/v1/query"
    echo ""
    echo "🔍 日志查看:"
    echo "  $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME logs -f [service_name]"
    echo "  $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME logs -f health-monitor"
}

# 停止服务
stop_services() {
    log_info "停止预投产环境服务..."

    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down

    log_success "服务已停止"
}

# 重启服务
restart_services() {
    log_info "重启预投产环境服务..."
    stop_services
    sleep 5
    start_services
    wait_for_services
    validate_deployment
}

# 显示状态
show_status() {
    log_info "预投产环境状态:"

    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps

    echo ""
    log_info "服务健康状态:"
    services=("postgres" "influxdb" "redis" "prometheus" "grafana" "elasticsearch" "kibana" "health-monitor")

    for service in "${services[@]}"; do
        if $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps "$service" | grep -q "healthy"; then
            echo -e "  ${GREEN}●${NC} $service: 健康"
        elif $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps "$service" | grep -q "running\|Up"; then
            echo -e "  ${YELLOW}●${NC} $service: 运行中"
        else
            echo -e "  ${RED}●${NC} $service: 停止"
        fi
    done
}

# 显示日志
show_logs() {
    service=${2:-"health-monitor"}

    log_info "显示 $service 服务日志:"

    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f "$service"
}

# 清理环境
cleanup() {
    log_warning "清理预投产环境..."

    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    # 停止并删除容器
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans

    # 删除镜像（可选）
    if [ "$1" = "--full" ]; then
        log_warning "删除相关Docker镜像..."
        docker images | grep "rqa2025" | awk '{print $3}' | xargs -r docker rmi
    fi

    # 删除环境文件
    if [ -f "$ENV_FILE" ]; then
        rm -f "$ENV_FILE"
        log_info "已删除环境配置文件"
    fi

    log_success "环境清理完成"
}

# 显示帮助
show_help() {
    echo "RQA2025 预投产环境部署工具"
    echo ""
    echo "使用方法:"
    echo "  $0 [command] [options]"
    echo ""
    echo "可用命令:"
    echo "  up         启动预投产环境"
    echo "  down       停止预投产环境"
    echo "  restart    重启预投产环境"
    echo "  status     显示环境状态"
    echo "  logs [svc] 显示服务日志 (默认: health-monitor)"
    echo "  validate   验证部署状态"
    echo "  info       显示服务信息"
    echo "  cleanup    清理环境 (--full 删除镜像)"
    echo "  help       显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 up          # 启动环境"
    echo "  $0 status      # 查看状态"
    echo "  $0 logs redis  # 查看Redis日志"
    echo "  $0 cleanup     # 清理环境"
}

# 主函数
main() {
    command=${1:-"help"}

    case $command in
        "up")
            check_dependencies
            create_env_file
            start_services
            wait_for_services
            validate_deployment
            show_service_info
            ;;
        "down")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$@"
            ;;
        "validate")
            validate_deployment
            ;;
        "info")
            show_service_info
            ;;
        "cleanup")
            cleanup "$2"
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"

