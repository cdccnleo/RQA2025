#!/bin/bash

# 自动更新测试覆盖率面板脚本
# 用法: ./auto_update_coverage_dashboard.sh [选项]

set -e

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
LOG_DIR="$PROJECT_ROOT/logs"
REPORTS_DIR="$PROJECT_ROOT/reports"

# 默认配置
UPDATE_INTERVAL=3600  # 1小时
DB_PATH="data/coverage_monitor.db"
OUTPUT_FILE="reports/coverage_dashboard.html"
LOG_FILE="$LOG_DIR/coverage_dashboard_auto_update.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $*${NC}" >&2 | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: $*${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S') - INFO: $*${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S') - WARNING: $*${NC}" | tee -a "$LOG_FILE"
}

# 创建必要的目录
setup_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$REPORTS_DIR"
    mkdir -p "$(dirname "$DB_PATH")"
}

# 更新面板
update_dashboard() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    info "开始更新覆盖率面板..."

    if [ ! -f "$DB_PATH" ]; then
        warning "数据库文件不存在: $DB_PATH"
        info "将使用内置数据生成面板..."
    fi

    # 运行面板生成脚本
    if cd "$PROJECT_ROOT" && python "$SCRIPT_DIR/generate_coverage_dashboard.py" \
        --db-path "$DB_PATH" \
        --output "$OUTPUT_FILE"; then

        success "覆盖率面板更新成功: $OUTPUT_FILE"
        return 0
    else
        error "覆盖率面板更新失败"
        return 1
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
自动更新测试覆盖率面板脚本

用法:
    $0 [选项]

选项:
    -i, --interval SECONDS    更新间隔(秒)，默认3600(1小时)
    -d, --db-path PATH        数据库路径，默认: data/coverage_monitor.db
    -o, --output PATH         输出文件路径，默认: reports/coverage_dashboard.html
    -l, --log-file PATH       日志文件路径，默认: logs/coverage_dashboard_auto_update.log
    -c, --continuous          连续运行模式
    -s, --single              单次执行模式(默认)
    -h, --help                显示此帮助信息

示例:
    # 单次更新
    $0 --single

    # 每30分钟自动更新
    $0 --continuous --interval 1800

    # 指定自定义路径
    $0 --db-path /path/to/db --output /path/to/dashboard.html

    # 在后台运行
    nohup $0 --continuous > /dev/null 2>&1 &
EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--interval)
                UPDATE_INTERVAL="$2"
                shift 2
                ;;
            -d|--db-path)
                DB_PATH="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -l|--log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            -c|--continuous)
                CONTINUOUS_MODE=true
                shift
                ;;
            -s|--single)
                CONTINUOUS_MODE=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 信号处理
cleanup() {
    info "收到停止信号，正在清理..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# 主函数
main() {
    local CONTINUOUS_MODE=false

    # 解析参数
    parse_args "$@"

    # 设置目录
    setup_directories

    info "=== 覆盖率面板自动更新服务启动 ==="
    info "项目根目录: $PROJECT_ROOT"
    info "数据库路径: $DB_PATH"
    info "输出文件: $OUTPUT_FILE"
    info "日志文件: $LOG_FILE"
    info "更新间隔: ${UPDATE_INTERVAL}秒"

    if [ "$CONTINUOUS_MODE" = true ]; then
        info "运行模式: 连续更新"
        while true; do
            if update_dashboard; then
                info "等待 ${UPDATE_INTERVAL} 秒后进行下次更新..."
                sleep "$UPDATE_INTERVAL"
            else
                warning "更新失败，${UPDATE_INTERVAL} 秒后重试..."
                sleep "$UPDATE_INTERVAL"
            fi
        done
    else
        info "运行模式: 单次执行"
        update_dashboard
        success "单次更新完成"
    fi
}

# 执行主函数
main "$@"
