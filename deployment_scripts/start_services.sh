#!/bin/bash
# RQA2026 服务启动脚本
# 用于生产环境启动所有服务组件

set -e  # 遇到错误立即退出

# 配置变量
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/pids"

# 创建必要的目录
mkdir -p "$LOG_DIR" "$PID_DIR"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/startup.log"
}

# 错误处理
error_exit() {
    log "ERROR: $1"
    exit 1
}

# 检查依赖
check_dependencies() {
    log "检查系统依赖..."

    # 检查Python
    if ! command -v python3 &> /dev/null; then
        error_exit "Python3 未找到"
    fi

    # 检查Node.js (用于Web界面)
    if ! command -v node &> /dev/null; then
        error_exit "Node.js 未找到"
    fi

    # 检查Redis
    if ! command -v redis-server &> /dev/null; then
        log "WARNING: Redis 未找到，将使用内存存储"
    fi

    # 检查PostgreSQL
    if ! command -v psql &> /dev/null; then
        error_exit "PostgreSQL 未找到"
    fi

    log "依赖检查完成"
}

# 启动数据库
start_database() {
    log "启动 PostgreSQL 数据库..."

    # 检查数据库是否已经在运行
    if pg_isready -h localhost -p 5432 &> /dev/null; then
        log "数据库已在运行"
        return 0
    fi

    # 启动PostgreSQL服务
    sudo systemctl start postgresql || error_exit "无法启动 PostgreSQL"

    # 等待数据库启动
    timeout=30
    while [ $timeout -gt 0 ]; do
        if pg_isready -h localhost -p 5432 &> /dev/null; then
            log "数据库启动成功"
            return 0
        fi
        sleep 1
        timeout=$((timeout - 1))
    done

    error_exit "数据库启动超时"
}

# 启动Redis缓存
start_redis() {
    log "启动 Redis 缓存服务..."

    if command -v redis-server &> /dev/null; then
        if pgrep -x "redis-server" > /dev/null; then
            log "Redis 已在运行"
        else
            redis-server --daemonize yes --port 6379
            log "Redis 启动成功"
        fi
    else
        log "Redis 未安装，使用内存缓存"
    fi
}

# 启动量子计算引擎
start_quantum_engine() {
    log "启动量子计算引擎..."

    cd "$PROJECT_ROOT/quantum_research/engine"

    # 检查端口是否被占用
    if lsof -Pi :8081 -sTCP:LISTEN -t >/dev/null; then
        log "端口 8081 已被占用"
        return 1
    fi

    # 启动服务
    nohup python3 -m uvicorn quantum_engine:app \
        --host 0.0.0.0 \
        --port 8081 \
        --workers 2 \
        --log-level info \
        > "$LOG_DIR/quantum_engine.log" 2>&1 &
    echo $! > "$PID_DIR/quantum_engine.pid"

    log "量子引擎启动完成 (PID: $(cat "$PID_DIR/quantum_engine.pid"))"
}

# 启动AI引擎
start_ai_engine() {
    log "启动 AI 深度集成引擎..."

    cd "$PROJECT_ROOT/multimodal_ai/engine"

    if lsof -Pi :8082 -sTCP:LISTEN -t >/dev/null; then
        log "端口 8082 已被占用"
        return 1
    fi

    nohup python3 -m uvicorn ai_engine:app \
        --host 0.0.0.0 \
        --port 8082 \
        --workers 4 \
        --log-level info \
        > "$LOG_DIR/ai_engine.log" 2>&1 &
    echo $! > "$PID_DIR/ai_engine.pid"

    log "AI引擎启动完成 (PID: $(cat "$PID_DIR/ai_engine.pid"))"
}

# 启动BCI引擎
start_bci_engine() {
    log "启动脑机接口引擎..."

    cd "$PROJECT_ROOT/bmi_research/engine"

    if lsof -Pi :8083 -sTCP:LISTEN -t >/dev/null; then
        log "端口 8083 已被占用"
        return 1
    fi

    nohup python3 -m uvicorn bci_engine:app \
        --host 0.0.0.0 \
        --port 8083 \
        --workers 2 \
        --log-level info \
        > "$LOG_DIR/bci_engine.log" 2>&1 &
    echo $! > "$PID_DIR/bci_engine.pid"

    log "BCI引擎启动完成 (PID: $(cat "$PID_DIR/bci_engine.pid"))"
}

# 启动融合引擎
start_fusion_engine() {
    log "启动融合引擎..."

    cd "$PROJECT_ROOT/innovation_fusion/architecture"

    if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null; then
        log "端口 8080 已被占用"
        return 1
    fi

    nohup python3 -m uvicorn fusion_engine:app \
        --host 0.0.0.0 \
        --port 8080 \
        --workers 2 \
        --log-level info \
        > "$LOG_DIR/fusion_engine.log" 2>&1 &
    echo $! > "$PID_DIR/fusion_engine.pid"

    log "融合引擎启动完成 (PID: $(cat "$PID_DIR/fusion_engine.pid"))"
}

# 启动Web界面
start_web_interface() {
    log "启动 Web 界面..."

    cd "$PROJECT_ROOT/web_interface"

    if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null; then
        log "端口 3000 已被占用"
        return 1
    fi

    # 检查是否是Node.js项目
    if [ -f "package.json" ]; then
        npm install
        nohup npm start > "$LOG_DIR/web_interface.log" 2>&1 &
    else
        # 如果没有Web界面，使用简单的Flask应用
        cd "$PROJECT_ROOT"
        nohup python3 -c "
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <h1>RQA2026 创新引擎平台</h1>
    <p>三大引擎运行状态:</p>
    <ul>
        <li>量子引擎: <a href=\"http://localhost:8081\">http://localhost:8081</a></li>
        <li>AI引擎: <a href=\"http://localhost:8082\">http://localhost:8082</a></li>
        <li>BCI引擎: <a href=\"http://localhost:8083\">http://localhost:8083</a></li>
        <li>融合引擎: <a href=\"http://localhost:8080\">http://localhost:8080</a></li>
    </ul>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
" > "$LOG_DIR/web_interface.log" 2>&1 &
    fi

    echo $! > "$PID_DIR/web_interface.pid"
    log "Web界面启动完成 (PID: $(cat "$PID_DIR/web_interface.pid"))"
}

# 启动监控系统
start_monitoring() {
    log "启动监控系统..."

    # 启动Prometheus (如果已安装)
    if command -v prometheus &> /dev/null; then
        if pgrep -x "prometheus" > /dev/null; then
            log "Prometheus 已在运行"
        else
            nohup prometheus --config.file="$PROJECT_ROOT/deployment_scripts/prometheus.yml" \
                > "$LOG_DIR/prometheus.log" 2>&1 &
            echo $! > "$PID_DIR/prometheus.pid"
            log "Prometheus 启动成功"
        fi
    fi

    # 启动Grafana (如果已安装)
    if command -v grafana-server &> /dev/null; then
        if pgrep -x "grafana-server" > /dev/null; then
            log "Grafana 已在运行"
        else
            nohup grafana-server --config="$PROJECT_ROOT/deployment_scripts/grafana.ini" \
                > "$LOG_DIR/grafana.log" 2>&1 &
            echo $! > "$PID_DIR/grafana.pid"
            log "Grafana 启动成功"
        fi
    fi
}

# 健康检查
health_check() {
    log "执行服务健康检查..."

    services=(
        "quantum_engine:8081"
        "ai_engine:8082"
        "bci_engine:8083"
        "fusion_engine:8080"
        "web_interface:3000"
    )

    for service in "${services[@]}"; do
        name="${service%%:*}"
        port="${service##*:}"

        if curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
            log "$name 健康检查通过"
        else
            log "WARNING: $name 健康检查失败"
        fi
    done
}

# 主函数
main() {
    log "开始 RQA2026 服务启动流程..."

    # 检查依赖
    check_dependencies

    # 启动基础服务
    start_database
    start_redis

    # 启动核心引擎
    start_quantum_engine
    start_ai_engine
    start_bci_engine
    start_fusion_engine

    # 启动用户界面
    start_web_interface

    # 启动监控
    start_monitoring

    # 等待服务启动
    log "等待服务完全启动..."
    sleep 10

    # 健康检查
    health_check

    log "RQA2026 平台启动完成！"
    log "访问地址:"
    log "  Web界面: http://localhost:3000"
    log "  融合API: http://localhost:8080"
    log "  监控面板: http://localhost:3001 (如已配置)"

    # 保存启动状态
    echo "STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')" > "$PROJECT_ROOT/.startup_status"
}

# 停止服务函数
stop_services() {
    log "停止所有服务..."

    # 读取PID文件并停止进程
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                log "停止进程 $pid ($(basename "$pid_file" .pid))"
            fi
            rm "$pid_file"
        fi
    done

    # 停止系统服务
    sudo systemctl stop postgresql 2>/dev/null || true

    log "所有服务已停止"
}

# 重启服务函数
restart_services() {
    log "重启所有服务..."
    stop_services
    sleep 3
    main
}

# 根据参数执行不同操作
case "${1:-start}" in
    start)
        main
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        # 显示服务状态
        echo "RQA2026 服务状态:"
        for pid_file in "$PID_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                service_name=$(basename "$pid_file" .pid)
                if kill -0 "$pid" 2>/dev/null; then
                    echo "  $service_name: 运行中 (PID: $pid)"
                else
                    echo "  $service_name: 已停止"
                fi
            fi
        done
        ;;
    health)
        health_check
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|health}"
        exit 1
        ;;
esac
