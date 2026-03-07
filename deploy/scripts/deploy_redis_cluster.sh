#!/bin/bash

# Redis集群部署脚本
# 使用方法: ./deploy_redis_cluster.sh

set -e

echo "开始部署Redis集群..."

# 配置变量
REDIS_NODES=(
    "192.168.1.10:6379"
    "192.168.1.11:6379"
    "192.168.1.12:6379"
    "192.168.1.13:6379"
    "192.168.1.14:6379"
    "192.168.1.15:6379"
)

# 检查Redis是否已安装
check_redis_installation() {
    echo "检查Redis安装..."
    if ! command -v redis-server &> /dev/null; then
        echo "Redis未安装，开始安装..."
        sudo apt update
        sudo apt install -y redis-server
    else
        echo "Redis已安装"
    fi
}

# 配置Redis集群
configure_redis_cluster() {
    echo "配置Redis集群..."
    
    # 备份原配置
    sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup
    
    # 创建集群配置
    cat > /tmp/redis-cluster.conf << EOF
# Redis集群配置
port 6379
bind 0.0.0.0
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
save 900 1
save 300 10
save 60 10000
maxmemory 2gb
maxmemory-policy allkeys-lru
EOF
    
    # 应用配置
    sudo cp /tmp/redis-cluster.conf /etc/redis/redis.conf
    sudo systemctl restart redis-server
    
    echo "Redis集群配置完成"
}

# 创建集群
create_cluster() {
    echo "创建Redis集群..."
    
    # 构建节点参数
    NODE_PARAMS=""
    for node in "${REDIS_NODES[@]}"; do
        NODE_PARAMS="$NODE_PARAMS $node"
    done
    
    # 创建集群
    redis-cli --cluster create $NODE_PARAMS --cluster-replicas 1 --cluster-yes
    
    echo "Redis集群创建完成"
}

# 验证集群状态
verify_cluster() {
    echo "验证集群状态..."
    
    # 检查集群信息
    redis-cli -h 192.168.1.10 -p 6379 cluster info
    
    # 检查节点状态
    redis-cli -h 192.168.1.10 -p 6379 cluster nodes
    
    echo "集群验证完成"
}

# 测试集群连接
test_cluster() {
    echo "测试集群连接..."
    
    # 测试写入
    redis-cli -h 192.168.1.10 -p 6379 set test_key "test_value"
    
    # 测试读取
    redis-cli -h 192.168.1.11 -p 6379 get test_key
    
    # 清理测试数据
    redis-cli -h 192.168.1.10 -p 6379 del test_key
    
    echo "集群连接测试完成"
}

# 主函数
main() {
    echo "=== Redis集群部署脚本 ==="
    
    check_redis_installation
    configure_redis_cluster
    create_cluster
    verify_cluster
    test_cluster
    
    echo "Redis集群部署完成！"
    echo "集群节点: ${REDIS_NODES[*]}"
}

# 执行主函数
main "$@" 