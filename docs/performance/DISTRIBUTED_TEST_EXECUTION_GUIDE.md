# 分布式测试执行框架指南

## 概述

本指南介绍如何使用RQA2025分布式测试执行框架，实现多机分布式测试执行，提高大规模测试的执行效率。

## 核心特性

### 1. 主从架构
- **主节点 (Master)**: 负责任务分发、结果收集和集群管理
- **工作节点 (Worker)**: 负责测试执行和结果上报
- **进程间通信**: 使用multiprocessing和socket进行高效通信

### 2. 智能任务分发
- **轮询分发 (Round Robin)**: 按顺序轮流分配给工作节点
- **负载均衡 (Load Balanced)**: 根据节点负载智能分配
- **随机分发 (Random)**: 随机选择工作节点

### 3. 集群管理
- **节点状态监控**: 实时监控工作节点状态
- **心跳检测**: 定期检测节点存活状态
- **故障容错**: 支持节点故障检测和恢复

## 架构组件

### DistributedTestRunner
分布式测试执行器的主控制器，负责：
- 启动和停止主节点
- 管理工作节点
- 分发测试任务
- 收集测试结果
- 监控集群状态

### WorkerNode
工作节点实现，负责：
- 接收测试任务
- 执行测试
- 上报结果
- 心跳维护

### NodeInfo
节点信息数据结构，包含：
- 节点ID、主机、端口
- 角色和状态
- 能力和负载信息
- 心跳时间戳

## 快速开始

### 1. 基本配置

```python
from src.infrastructure.performance.distributed_test_runner import (
    DistributedTestConfig,
    create_distributed_test_runner
)

# 创建配置
config = DistributedTestConfig(
    master_host="localhost",
    master_port=5000,
    worker_timeout=30,
    heartbeat_interval=5,
    max_workers_per_node=4,
    test_distribution_strategy="load_balanced"
)

# 创建分布式测试运行器
runner = create_distributed_test_runner(config)
```

### 2. 添加工作节点

```python
# 添加工作节点
capabilities = {
    'max_workers': 4,
    'memory_gb': 8,
    'cpu_cores': 2
}

runner.add_worker_node(
    node_id="worker-001",
    host="192.168.1.101",
    port=8081,
    capabilities=capabilities
)

runner.add_worker_node(
    node_id="worker-002",
    host="192.168.1.102",
    port=8082,
    capabilities=capabilities
)
```

### 3. 启动主节点

```python
# 启动主节点
runner.start_master()

# 检查集群状态
status = runner.get_cluster_status()
print(f"集群状态: {status}")
```

### 4. 分发和执行测试

```python
# 定义测试套件
def test_function1():
    return "result1"

def test_function2():
    return "result2"

test_suite = [
    ("test1", test_function1, {}),
    ("test2", test_function2, {})
]

# 分发测试
runner.distribute_tests(test_suite)

# 收集结果
results = runner.collect_results(timeout=300)
print(f"收集到 {len(results)} 个测试结果")
```

### 5. 停止主节点

```python
# 停止主节点
runner.stop_master()
```

## 高级配置

### 1. 分发策略配置

```python
# 轮询分发
config.test_distribution_strategy = "round_robin"

# 负载均衡
config.test_distribution_strategy = "load_balanced"

# 随机分发
config.test_distribution_strategy = "random"
```

### 2. 故障容错配置

```python
config = DistributedTestConfig(
    enable_fault_tolerance=True,
    retry_failed_tests=True,
    max_retries=3,
    worker_timeout=60
)
```

### 3. 性能优化配置

```python
config = DistributedTestConfig(
    max_workers_per_node=8,
    heartbeat_interval=3,
    worker_timeout=120
)
```

## 工作节点部署

### 1. 独立工作节点

```python
from src.infrastructure.performance.distributed_test_runner import (
    create_worker_node
)

# 创建工作节点
worker = create_worker_node(
    node_id="worker-001",
    master_host="192.168.1.100",
    master_port=5000,
    capabilities={'max_workers': 4}
)

# 启动工作节点
worker.start()

# 保持运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    worker.stop()
```

### 2. 集群部署

```python
# 部署多个工作节点
worker_nodes = [
    {
        'node_id': 'worker-001',
        'host': '192.168.1.101',
        'port': 8081,
        'capabilities': {'max_workers': 4}
    },
    {
        'node_id': 'worker-002',
        'host': '192.168.1.102',
        'port': 8082,
        'capabilities': {'max_workers': 6}
    },
    {
        'node_id': 'worker-003',
        'host': '192.168.1.103',
        'port': 8083,
        'capabilities': {'max_workers': 8}
    }
]

# 批量添加工作节点
for worker_info in worker_nodes:
    runner.add_worker_node(**worker_info)
```

## 监控和调试

### 1. 集群状态监控

```python
# 获取集群状态
status = runner.get_cluster_status()

print(f"主节点状态: {status['master_status']}")
print(f"总工作节点数: {status['total_workers']}")
print(f"在线工作节点: {status['online_workers']}")
print(f"忙碌工作节点: {status['busy_workers']}")
print(f"空闲工作节点: {status['idle_workers']}")

# 测试进度
progress = status['test_progress']
print(f"测试进度: {progress['completed']}/{progress['total']}")
print(f"失败测试: {progress['failed']}")
print(f"执行时间: {status['execution_time']:.2f}s")
```

### 2. 节点状态监控

```python
# 检查特定节点状态
for node_id, node_info in runner.worker_nodes.items():
    print(f"节点 {node_id}:")
    print(f"  状态: {node_info.status.value}")
    print(f"  负载: {node_info.load:.2f}")
    print(f"  最后心跳: {time.time() - node_info.last_heartbeat:.1f}s前")
```

### 3. 日志配置

```python
import logging

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 分布式测试运行器日志
logger = logging.getLogger('src.infrastructure.performance.distributed_test_runner')
logger.setLevel(logging.DEBUG)
```

## 最佳实践

### 1. 节点配置

- **合理分配资源**: 根据节点硬件配置设置max_workers
- **网络优化**: 确保主节点和工作节点间网络延迟低
- **负载均衡**: 使用load_balanced策略实现最佳性能

### 2. 测试设计

- **可序列化**: 确保测试函数可以被pickle序列化
- **独立性**: 测试之间应该相互独立，避免状态共享
- **资源管理**: 合理管理测试资源，避免内存泄漏

### 3. 集群管理

- **监控告警**: 实现节点故障检测和告警机制
- **自动恢复**: 支持工作节点自动重启和恢复
- **容量规划**: 根据测试负载合理规划集群规模

## 故障排除

### 1. 常见问题

**问题**: 工作节点无法连接主节点
```python
# 解决方案：检查网络配置
# 1. 验证防火墙设置
# 2. 检查端口是否开放
# 3. 确认主机名解析
```

**问题**: 测试任务分发失败
```python
# 解决方案：检查工作节点状态
# 1. 确保工作节点在线
# 2. 检查节点负载
# 3. 验证分发策略配置
```

**问题**: 结果收集超时
```python
# 解决方案：调整超时配置
config = DistributedTestConfig(
    worker_timeout=120,  # 增加超时时间
    heartbeat_interval=3  # 减少心跳间隔
)
```

### 2. 性能调优

```python
# 优化配置示例
config = DistributedTestConfig(
    max_workers_per_node=8,      # 增加工作线程
    heartbeat_interval=2,        # 减少心跳间隔
    worker_timeout=180,          # 增加超时时间
    test_distribution_strategy="load_balanced"  # 使用负载均衡
)
```

## 总结

通过本指南，您可以成功使用RQA2025分布式测试执行框架，实现：

- **高效测试执行**: 多机并行执行，显著提升测试效率
- **智能任务分发**: 多种分发策略，优化资源利用
- **可靠集群管理**: 完善的监控和故障处理机制
- **灵活部署**: 支持各种部署场景和配置需求

遵循最佳实践，合理配置集群，您将能够构建一个高效、可靠的分布式测试执行环境。
