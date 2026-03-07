#!/bin/bash

# RQA2025 基础设施准备脚本
# 使用方法: ./prepare_infrastructure.sh

set -e

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

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "操作系统: Linux"
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    # 检查内存
    local total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    if [ "$total_mem" -lt 8 ]; then
        log_error "内存不足，需要至少8GB，当前: ${total_mem}GB"
        exit 1
    else
        log_info "内存检查通过: ${total_mem}GB"
    fi
    
    # 检查磁盘空间
    local free_space=$(df -BG / | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$free_space" -lt 50 ]; then
        log_error "磁盘空间不足，需要至少50GB，当前: ${free_space}GB"
        exit 1
    else
        log_info "磁盘空间检查通过: ${free_space}GB"
    fi
    
    # 检查CPU核心数
    local cpu_cores=$(nproc)
    if [ "$cpu_cores" -lt 4 ]; then
        log_warn "CPU核心数较少: ${cpu_cores}，建议至少4核"
    else
        log_info "CPU核心数检查通过: ${cpu_cores}核"
    fi
}

# 检查网络连通性
check_network_connectivity() {
    log_info "检查网络连通性..."
    
    # 检查外网连接
    if ping -c 1 8.8.8.8 &> /dev/null; then
        log_info "外网连接正常"
    else
        log_warn "外网连接异常"
    fi
    
    # 检查DNS解析
    if nslookup google.com &> /dev/null; then
        log_info "DNS解析正常"
    else
        log_warn "DNS解析异常"
    fi
    
    # 检查端口可用性
    local ports=(22 80 443 5432 6379 9090 3000)
    for port in "${ports[@]}"; do
        if netstat -tuln | grep ":$port " &> /dev/null; then
            log_warn "端口 $port 已被占用"
        else
            log_info "端口 $port 可用"
        fi
    done
}

# 安装系统依赖
install_system_dependencies() {
    log_info "安装系统依赖..."
    
    # 更新包管理器
    sudo apt update
    
    # 安装基础工具
    sudo apt install -y \
        curl \
        wget \
        git \
        vim \
        htop \
        net-tools \
        unzip \
        software-properties-common
    
    # 安装Docker
    if ! command -v docker &> /dev/null; then
        log_info "安装Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    else
        log_info "Docker已安装"
    fi
    
    # 安装Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_info "安装Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        log_info "Docker Compose已安装"
    fi
    
    # 安装Python依赖
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential
}

# 创建目录结构
create_directory_structure() {
    log_info "创建目录结构..."
    
    # 创建应用目录
    sudo mkdir -p /opt/rqa2025/{config,logs,data,models,cache}
    sudo mkdir -p /var/log/rqa2025
    sudo mkdir -p /etc/rqa2025
    
    # 创建监控目录
    sudo mkdir -p /opt/monitoring/{prometheus,grafana,alertmanager}
    sudo mkdir -p /var/lib/prometheus
    sudo mkdir -p /var/lib/grafana
    
    # 创建日志目录
    sudo mkdir -p /opt/logging/{elasticsearch,logstash,kibana}
    sudo mkdir -p /var/lib/elasticsearch
    
    # 设置权限
    sudo chown -R $USER:$USER /opt/rqa2025
    sudo chown -R $USER:$USER /var/log/rqa2025
    sudo chown -R $USER:$USER /opt/monitoring
    sudo chown -R $USER:$USER /opt/logging
    
    log_info "目录结构创建完成"
}

# 配置防火墙
configure_firewall() {
    log_info "配置防火墙..."
    
    # 安装ufw
    sudo apt install -y ufw
    
    # 设置默认策略
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # 允许SSH
    sudo ufw allow ssh
    
    # 允许HTTP/HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # 允许应用端口
    sudo ufw allow 8000/tcp  # API服务
    sudo ufw allow 8001/tcp  # 推理引擎
    sudo ufw allow 5432/tcp  # PostgreSQL
    sudo ufw allow 6379/tcp  # Redis
    
    # 允许监控端口
    sudo ufw allow 9090/tcp  # Prometheus
    sudo ufw allow 3000/tcp  # Grafana
    sudo ufw allow 9093/tcp  # AlertManager
    sudo ufw allow 9200/tcp  # Elasticsearch
    
    # 启用防火墙
    sudo ufw --force enable
    
    log_info "防火墙配置完成"
}

# 配置系统参数
configure_system_parameters() {
    log_info "配置系统参数..."
    
    # 增加文件描述符限制
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
    
    # 增加进程数限制
    echo "* soft nproc 32768" | sudo tee -a /etc/security/limits.conf
    echo "* hard nproc 32768" | sudo tee -a /etc/security/limits.conf
    
    # 配置内核参数
    echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
    echo "net.core.somaxconn=65535" | sudo tee -a /etc/sysctl.conf
    echo "net.ipv4.tcp_max_syn_backlog=65535" | sudo tee -a /etc/sysctl.conf
    
    # 应用内核参数
    sudo sysctl -p
    
    log_info "系统参数配置完成"
}

# 配置时区和NTP
configure_time_settings() {
    log_info "配置时区和时间同步..."
    
    # 设置时区
    sudo timedatectl set-timezone Asia/Shanghai
    
    # 安装NTP
    sudo apt install -y ntp
    
    # 启动NTP服务
    sudo systemctl enable ntp
    sudo systemctl start ntp
    
    log_info "时区和时间同步配置完成"
}

# 主函数
main() {
    log_info "开始基础设施准备..."
    
    check_system_requirements
    check_network_connectivity
    install_system_dependencies
    create_directory_structure
    configure_firewall
    configure_system_parameters
    configure_time_settings
    
    log_info "基础设施准备完成！"
    log_info "请重新登录以应用Docker组权限"
}

# 执行主函数
main "$@" 