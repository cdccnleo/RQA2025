#!/bin/bash
# RQA2025 生产环境部署脚本
# 版本: 1.0.0
# 创建日期: 2025-01-27

set -euo pipefail

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

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config/production"
DEPLOY_DIR="$PROJECT_ROOT/deploy"
BACKUP_DIR="$PROJECT_ROOT/backups"

# 部署配置
APP_NAME="rqa2025"
DEPLOYMENT_STRATEGY="blue-green"
HEALTH_CHECK_ENDPOINT="/health"
HEALTH_CHECK_TIMEOUT=30
HEALTH_CHECK_RETRIES=3

# 环境变量
export ENVIRONMENT="production"
export CONFIG_PATH="$CONFIG_DIR/config.yaml"

# 显示帮助信息
show_help() {
    cat << EOF
RQA2025 生产环境部署脚本

用法:
    $0 [选项] [命令]

命令:
    deploy      部署新版本
    rollback    回滚到上一个版本
    status      检查部署状态
    health      健康检查
    backup      创建备份
    validate    验证配置

选项:
    -h, --help          显示此帮助信息
    -v, --version       显示版本信息
    -f, --force         强制部署（跳过检查）
    -d, --dry-run       试运行模式
    -c, --config PATH   指定配置文件路径
    -t, --tag TAG       指定部署标签

示例:
    $0 deploy -t v1.0.0
    $0 rollback
    $0 status
    $0 health

EOF
}

# 显示版本信息
show_version() {
    echo "RQA2025 生产环境部署脚本 v1.0.0"
}

# 验证环境
validate_environment() {
    log_info "验证部署环境..."
    
    # 检查必要的命令
    local required_commands=("docker" "kubectl" "helm" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "缺少必要的命令: $cmd"
            exit 1
        fi
    done
    
    # 检查配置文件
    if [[ ! -f "$CONFIG_PATH" ]]; then
        log_error "配置文件不存在: $CONFIG_PATH"
        exit 1
    fi
    
    # 检查环境变量
    local required_vars=("DB_HOST" "DB_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "缺少必要的环境变量: $var"
            exit 1
        fi
    done
    
    log_success "环境验证通过"
}

# 创建备份
create_backup() {
    log_info "创建系统备份..."
    
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_name="backup_${backup_timestamp}"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    mkdir -p "$backup_path"
    
    # 备份配置文件
    cp -r "$CONFIG_DIR" "$backup_path/"
    
    # 备份数据库（如果可能）
    if command -v pg_dump &> /dev/null; then
        log_info "备份数据库..."
        pg_dump -h "$DB_HOST" -U "$DB_USERNAME" "$DB_NAME" > "$backup_path/database.sql"
    fi
    
    # 备份Docker镜像
    log_info "备份Docker镜像..."
    docker save "$APP_NAME:latest" > "$backup_path/image.tar"
    
    log_success "备份创建完成: $backup_path"
}

# 验证配置
validate_config() {
    log_info "验证配置文件..."
    
    # 验证YAML语法
    if command -v python3 &> /dev/null; then
        python3 -c "import yaml; yaml.safe_load(open('$CONFIG_PATH'))" || {
            log_error "配置文件YAML语法错误"
            exit 1
        }
    fi
    
    # 验证Kubernetes配置
    if [[ -f "$DEPLOY_DIR/kubernetes/deployment.yaml" ]]; then
        kubectl apply --dry-run=client -f "$DEPLOY_DIR/kubernetes/deployment.yaml" || {
            log_error "Kubernetes配置验证失败"
            exit 1
        }
    fi
    
    log_success "配置验证通过"
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    local endpoint="$HEALTH_CHECK_ENDPOINT"
    local timeout="$HEALTH_CHECK_TIMEOUT"
    local retries="$HEALTH_CHECK_RETRIES"
    
    for ((i=1; i<=retries; i++)); do
        log_info "健康检查尝试 $i/$retries..."
        
        if curl -f -s --max-time "$timeout" "http://localhost:8080$endpoint" > /dev/null; then
            log_success "健康检查通过"
            return 0
        else
            log_warning "健康检查失败，尝试 $i/$retries"
            sleep 5
        fi
    done
    
    log_error "健康检查失败"
    return 1
}

# 蓝绿部署
blue_green_deploy() {
    local new_version="$1"
    log_info "开始蓝绿部署，版本: $new_version"
    
    # 确定当前环境
    local current_env=""
    if kubectl get service "$APP_NAME" -o jsonpath='{.spec.selector.environment}' 2>/dev/null | grep -q "blue"; then
        current_env="blue"
        new_env="green"
    else
        current_env="green"
        new_env="blue"
    fi
    
    log_info "当前环境: $current_env, 新环境: $new_env"
    
    # 部署新版本到新环境
    log_info "部署新版本到 $new_env 环境..."
    kubectl apply -f "$DEPLOY_DIR/kubernetes/deployment-$new_env.yaml"
    
    # 等待新环境就绪
    log_info "等待新环境就绪..."
    kubectl rollout status deployment/"$APP_NAME-$new_env" --timeout=300s
    
    # 健康检查新环境
    log_info "健康检查新环境..."
    if health_check; then
        # 切换流量到新环境
        log_info "切换流量到新环境..."
        kubectl patch service "$APP_NAME" -p "{\"spec\":{\"selector\":{\"environment\":\"$new_env\"}}}"
        
        # 验证切换成功
        sleep 10
        if health_check; then
            log_success "蓝绿部署成功"
            # 清理旧环境
            log_info "清理旧环境..."
            kubectl delete deployment "$APP_NAME-$current_env"
        else
            log_error "流量切换失败，回滚..."
            kubectl patch service "$APP_NAME" -p "{\"spec\":{\"selector\":{\"environment\":\"$current_env\"}}}"
            exit 1
        fi
    else
        log_error "新环境健康检查失败"
        exit 1
    fi
}

# 滚动部署
rolling_deploy() {
    local new_version="$1"
    log_info "开始滚动部署，版本: $new_version"
    
    # 更新部署
    kubectl set image deployment/"$APP_NAME" "$APP_NAME=$APP_NAME:$new_version"
    
    # 等待部署完成
    log_info "等待部署完成..."
    kubectl rollout status deployment/"$APP_NAME" --timeout=300s
    
    # 健康检查
    if health_check; then
        log_success "滚动部署成功"
    else
        log_error "部署后健康检查失败"
        kubectl rollout undo deployment/"$APP_NAME"
        exit 1
    fi
}

# 部署主函数
deploy() {
    local version="$1"
    local strategy="$2"
    
    log_info "开始部署，版本: $version, 策略: $strategy"
    
    # 创建备份
    create_backup
    
    # 验证配置
    validate_config
    
    # 根据策略执行部署
    case "$strategy" in
        "blue-green")
            blue_green_deploy "$version"
            ;;
        "rolling")
            rolling_deploy "$version"
            ;;
        *)
            log_error "不支持的部署策略: $strategy"
            exit 1
            ;;
    esac
    
    log_success "部署完成"
}

# 回滚部署
rollback() {
    log_info "开始回滚部署..."
    
    # 获取上一个版本
    local previous_version=$(kubectl rollout history deployment/"$APP_NAME" --output=json | jq -r '.revisions[-2].revision')
    
    if [[ -z "$previous_version" ]]; then
        log_error "没有可回滚的版本"
        exit 1
    fi
    
    log_info "回滚到版本: $previous_version"
    
    # 执行回滚
    kubectl rollout undo deployment/"$APP_NAME" --to-revision="$previous_version"
    
    # 等待回滚完成
    kubectl rollout status deployment/"$APP_NAME" --timeout=300s
    
    # 健康检查
    if health_check; then
        log_success "回滚成功"
    else
        log_error "回滚后健康检查失败"
        exit 1
    fi
}

# 检查部署状态
status() {
    log_info "检查部署状态..."
    
    echo "=== 部署状态 ==="
    kubectl get deployments -l app="$APP_NAME"
    
    echo -e "\n=== 服务状态 ==="
    kubectl get services -l app="$APP_NAME"
    
    echo -e "\n=== Pod状态 ==="
    kubectl get pods -l app="$APP_NAME"
    
    echo -e "\n=== 事件 ==="
    kubectl get events --sort-by='.lastTimestamp' | tail -10
    
    echo -e "\n=== 资源使用 ==="
    kubectl top pods -l app="$APP_NAME"
}

# 主函数
main() {
    local command=""
    local version="latest"
    local strategy="$DEPLOYMENT_STRATEGY"
    local force=false
    local dry_run=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -d|--dry-run)
                dry_run=true
                shift
                ;;
            -c|--config)
                CONFIG_PATH="$2"
                shift 2
                ;;
            -t|--tag)
                version="$2"
                shift 2
                ;;
            deploy|rollback|status|health|backup|validate)
                command="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查命令
    if [[ -z "$command" ]]; then
        log_error "请指定命令"
        show_help
        exit 1
    fi
    
    # 验证环境（除了status和health命令）
    if [[ "$command" != "status" && "$command" != "health" ]]; then
        validate_environment
    fi
    
    # 执行命令
    case "$command" in
        deploy)
            if [[ "$dry_run" == true ]]; then
                log_info "试运行模式 - 将部署版本: $version, 策略: $strategy"
            else
                deploy "$version" "$strategy"
            fi
            ;;
        rollback)
            if [[ "$dry_run" == true ]]; then
                log_info "试运行模式 - 将执行回滚"
            else
                rollback
            fi
            ;;
        status)
            status
            ;;
        health)
            health_check
            ;;
        backup)
            create_backup
            ;;
        validate)
            validate_config
            ;;
        *)
            log_error "未知命令: $command"
            exit 1
            ;;
    esac
}

# 错误处理
trap 'log_error "部署脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"
