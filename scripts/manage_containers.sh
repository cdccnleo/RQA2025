#!/bin/bash
# RQA2025 容器管理脚本
# 用于在构建和部署过程中管理Docker容器

set -e

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

# 检查容器状态
check_container_status() {
    local container_name=$1

    if docker ps | grep -q "$container_name"; then
        echo "running"
    elif docker ps -a | grep -q "$container_name"; then
        echo "stopped"
    else
        echo "not_exists"
    fi
}

# 停止容器
stop_container() {
    local container_name=$1
    local status=$(check_container_status "$container_name")

    case $status in
        "running")
            log_info "停止容器 $container_name..."
            docker stop "$container_name"
            log_success "容器 $container_name 已停止"
            ;;
        "stopped")
            log_info "容器 $container_name 已停止，无需操作"
            ;;
        "not_exists")
            log_info "容器 $container_name 不存在，无需操作"
            ;;
    esac
}

# 移除容器
remove_container() {
    local container_name=$1
    local status=$(check_container_status "$container_name")

    case $status in
        "running")
            log_warn "容器 $container_name 正在运行，需要先停止"
            stop_container "$container_name"
            remove_container "$container_name"
            ;;
        "stopped")
            log_info "移除容器 $container_name..."
            docker rm "$container_name"
            log_success "容器 $container_name 已移除"
            ;;
        "not_exists")
            log_info "容器 $container_name 不存在，无需操作"
            ;;
    esac
}

# 清理容器（停止并移除）
cleanup_container() {
    local container_name=$1

    log_info "清理容器 $container_name..."
    stop_container "$container_name"
    remove_container "$container_name"
    log_success "容器 $container_name 清理完成"
}

# 等待容器健康
wait_container_healthy() {
    local container_name=$1
    local max_attempts=30
    local attempt=1

    log_info "等待容器 $container_name 变为健康状态..."

    while [ $attempt -le $max_attempts ]; do
        if docker ps | grep -q "$container_name.*healthy"; then
            log_success "容器 $container_name 已健康"
            return 0
        fi

        log_info "等待中... ($attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done

    log_error "容器 $container_name 未在预期时间内变为健康状态"
    return 1
}

# 显示容器状态
show_container_status() {
    log_info "当前容器状态:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# 主函数
main() {
    local action=$1
    local container_name=${2:-rqa2025-app}

    case $action in
        "status")
            show_container_status
            ;;
        "stop")
            stop_container "$container_name"
            ;;
        "remove")
            remove_container "$container_name"
            ;;
        "cleanup")
            cleanup_container "$container_name"
            ;;
        "wait-healthy")
            wait_container_healthy "$container_name"
            ;;
        "pre-build")
            log_info "准备构建环境..."
            cleanup_container "$container_name"
            log_success "构建环境准备完成"
            ;;
        *)
            echo "用法: $0 {status|stop|remove|cleanup|wait-healthy|pre-build} [container_name]"
            echo ""
            echo "命令说明:"
            echo "  status        显示所有容器状态"
            echo "  stop          停止指定容器 (默认: rqa2025-app)"
            echo "  remove        移除指定容器 (默认: rqa2025-app)"
            echo "  cleanup       停止并移除指定容器 (默认: rqa2025-app)"
            echo "  wait-healthy  等待容器变为健康状态 (默认: rqa2025-app)"
            echo "  pre-build     构建前清理 (停止并移除 rqa2025-app)"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"