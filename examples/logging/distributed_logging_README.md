# 分布式Logger使用指南

## 📖 概述

RQA2025的分布式Logger系统提供了完整的分布式日志处理能力，包括多节点日志聚合、服务发现、配置同步和智能负载均衡。

## 🎯 核心特性

### 1. 多节点日志聚合
- **实时聚合**: 支持毫秒级日志聚合和处理
- **智能路由**: 基于负载和健康状态的智能日志路由
- **容错机制**: 节点故障时的自动故障转移和数据重放

### 2. 服务发现集成
- **多种实现**: 支持Consul、ZooKeeper等服务发现系统
- **自动注册**: 节点启动时自动注册到服务发现系统
- **健康检查**: 自动化的节点健康状态监控

### 3. 配置同步
- **跨节点同步**: 分布式环境下的配置一致性保证
- **版本控制**: 配置变更的版本管理和回滚
- **冲突解决**: 智能的配置冲突检测和解决

### 4. 智能负载均衡
- **多均衡策略**: 加权轮询、最少连接、自适应等多种策略
- **实时调整**: 基于节点负载和性能的实时权重调整
- **故障转移**: 节点故障时的自动负载重新分配

## 🚀 快速开始

### 1. 单节点部署

```python
from infrastructure.logging.distributed import DistributedLogCoordinator

# 创建单节点协调器
coordinator = DistributedLogCoordinator(
    node_id="node-1",
    host="localhost",
    port=8080
)

# 启动系统
coordinator.start()

# 提交日志
coordinator.submit_log_entry(
    level="INFO",
    message="系统启动",
    category="SYSTEM"
)

# 停止系统
coordinator.stop()
```

### 2. 多节点集群部署

```python
from infrastructure.logging.distributed import (
    DistributedLogCoordinator,
    ConsulServiceDiscovery,
    DistributedConfigSync,
    AdaptiveLoadBalancer
)

# 创建服务发现
discovery = ConsulServiceDiscovery(
    consul_host="consul.example.com",
    consul_port=8500
)

# 创建配置同步器
config_sync = DistributedConfigSync(
    node_id="node-1",
    sync_interval=30.0
)

# 创建负载均衡器
load_balancer = AdaptiveLoadBalancer(
    nodes=["node-1"],  # 初始节点
    adaptation_interval=60.0
)

# 创建集群协调器
coordinator = DistributedLogCoordinator(
    node_id="node-1",
    host="node1.example.com",
    port=8080,
    service_discovery=discovery,
    config_sync=config_sync,
    load_balancer=load_balancer
)

# 启动集群
coordinator.start()
```

### 3. 高可用配置

```python
# 启用高可用模式
coordinator.enable_high_availability()

# 配置一致性级别
coordinator.configure_consistency("quorum")

# 获取分布式状态
status = coordinator.get_distributed_status()
print(f"复制因子: {status['distributed_config']['replication_factor']}")
```

## 📋 详细配置

### 服务发现配置

#### Consul配置
```python
from infrastructure.logging.distributed import ConsulServiceDiscovery

discovery = ConsulServiceDiscovery(
    consul_host="consul.example.com",
    consul_port=8500,
    service_name="distributed-logger",
    heartbeat_interval=30.0,
    health_check_interval=10.0
)
```

#### ZooKeeper配置
```python
from infrastructure.logging.distributed import ZookeeperServiceDiscovery

discovery = ZookeeperServiceDiscovery(
    zk_hosts="zk1:2181,zk2:2181,zk3:2181",
    service_name="distributed-logger"
)
```

### 负载均衡配置

#### 自适应负载均衡
```python
from infrastructure.logging.distributed import AdaptiveLoadBalancer

balancer = AdaptiveLoadBalancer(
    nodes=["node-1", "node-2", "node-3"],
    adaptation_interval=60.0  # 每60秒调整一次权重
)
```

#### 加权负载均衡
```python
from infrastructure.logging.distributed import WeightedLoadBalancer

balancer = WeightedLoadBalancer(
    nodes=["node-1", "node-2", "node-3"],
    weights={
        "node-1": 2.0,  # 双倍权重
        "node-2": 1.0,  # 标准权重
        "node-3": 1.5   # 1.5倍权重
    }
)
```

### 配置同步配置

```python
from infrastructure.logging.distributed import DistributedConfigSync

config_sync = DistributedConfigSync(
    node_id="node-1",
    sync_interval=30.0,  # 每30秒同步一次
    conflict_resolution="last_write_wins"  # 最后写入胜出
)
```

## 🔧 高级功能

### 配置广播和同步

```python
# 广播配置更新
coordinator.broadcast_config_update("logger.level", "DEBUG")

# 手动触发配置同步
coordinator.config_sync.sync()

# 查看同步统计
stats = coordinator.config_sync.get_sync_stats()
print(f"同步成功率: {stats['successful_syncs']}/{stats['total_syncs']}")
```

### 负载均衡监控

```python
# 获取负载均衡统计
stats = coordinator.load_balancer.get_all_stats()

for node_id, node_stats in stats.items():
    print(f"节点 {node_id}:")
    print(f"  健康评分: {node_stats.health_score}")
    print(f"  利用率: {node_stats.utilization_rate:.2%}")
    print(f"  成功率: {node_stats.success_rate:.2%}")
```

### 集群状态监控

```python
# 获取集群状态
cluster_stats = coordinator.cluster_manager.get_cluster_stats()

print(f"集群节点数: {cluster_stats.total_nodes}")
print(f"活跃节点数: {cluster_stats.active_nodes}")
print(f"主节点: {coordinator.cluster_manager.master_node}")
```

### 日志聚合统计

```python
# 获取聚合统计
agg_stats = coordinator.aggregator.get_stats()

print(f"总日志条目: {agg_stats.total_entries}")
print(f"每秒处理量: {agg_stats.entries_per_second:.1f}")
print(f"缓冲区大小: {agg_stats.buffer_size}")
```

## 🛠️ 故障排除

### 常见问题

1. **节点无法加入集群**
   ```python
   # 检查服务发现配置
   discovery_status = coordinator.service_discovery.get_stats()
   print(f"服务实例数: {discovery_status['total_instances']}")

   # 检查网络连接
   import socket
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   result = sock.connect_ex((consul_host, consul_port))
   print(f"Consul连接: {'成功' if result == 0 else '失败'}")
   ```

2. **配置同步延迟**
   ```python
   # 检查同步统计
   sync_stats = coordinator.config_sync.get_sync_stats()
   print(f"最后同步时间: {sync_stats['last_sync_time']}")

   # 手动触发同步
   coordinator.config_sync.sync()
   ```

3. **负载均衡不均衡**
   ```python
   # 检查节点健康状态
   for node_id, stats in coordinator.load_balancer.get_all_stats().items():
       print(f"节点 {node_id} 健康评分: {stats.health_score}")

   # 重新平衡负载
   coordinator.rebalance_cluster()
   ```

## 📊 性能优化

### 优化建议

1. **调整同步间隔**
   ```python
   # 降低同步频率以减少网络开销
   config_sync.sync_interval = 60.0  # 改为60秒
   ```

2. **配置合适的缓冲区大小**
   ```python
   # 根据负载调整聚合器缓冲区
   aggregator.buffer_size = 50000  # 增加缓冲区
   ```

3. **选择合适的负载均衡策略**
   ```python
   # 高负载场景使用自适应策略
   load_balancer = AdaptiveLoadBalancer(nodes=nodes)
   ```

## 🔒 安全配置

### 启用加密通信

```python
# 配置TLS加密
coordinator.enable_encryption(
    cert_file="path/to/cert.pem",
    key_file="path/to/key.pem"
)
```

### 配置访问控制

```python
# 设置访问控制列表
coordinator.set_access_control({
    "allowed_nodes": ["node-1", "node-2"],
    "admin_nodes": ["node-1"],
    "read_only_nodes": ["node-3"]
})
```

## 📈 监控和告警

### 设置监控阈值

```python
# 配置监控阈值
coordinator.set_monitoring_thresholds({
    "max_response_time": 100,  # ms
    "min_success_rate": 0.95,  # 95%
    "max_queue_size": 10000,
    "min_healthy_nodes": 2
})
```

### 配置告警规则

```python
# 设置告警规则
coordinator.set_alert_rules({
    "node_failure": {
        "enabled": True,
        "channels": ["email", "slack"],
        "threshold": 1  # 1个节点失败即告警
    },
    "high_latency": {
        "enabled": True,
        "channels": ["webhook"],
        "threshold": 200  # 200ms
    }
})
```

## 📚 相关文档

- [Logger配置系统](../../docs/api/logger_api.md) - 配置选项详解
- [架构设计文档](../../docs/architecture/infrastructure_architecture_design.md) - 系统架构说明
- [重构完成报告](../../INFRASTRUCTURE_REFACTORING_COMPLETION_REPORT.md) - 重构详情

---

**分布式Logger系统**: 为高可用、大规模的日志处理场景提供企业级的分布式解决方案，支持跨数据中心的日志聚合和智能调度。
