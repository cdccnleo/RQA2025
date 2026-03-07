# 监控配置指南

## 概述

本文档详细说明了RQA2025系统监控模块的配置方法，包括动态配置调整、监控指标收集和最佳实践。

## 配置结构

### 基础配置

```python
monitoring_config = {
    "interval": 5,  # 监控采集间隔（秒）
    "storage": ["prometheus", "file"],  # 存储后端
    "prometheus": {
        "endpoint": "http://localhost:9090",
        "push_interval": 15
    },
    "influxdb": {
        "url": "http://localhost:8086",
        "database": "rqa_metrics",
        "username": "admin",
        "password": "password"
    },
    "file": {
        "path": "logs/performance.log",
        "max_size": "100MB",
        "backup_count": 5
    }
}
```

## 动态配置调整

### 运行时修改采集间隔

```python
from src.infrastructure.monitoring.performance_monitor import PerformanceMonitor

# 初始化监控器
monitor = PerformanceMonitor(config)

# 动态调整采集间隔
monitor.set_config({'interval': 10})

# 或者更新单个配置项
monitor.update_config('interval', 15)
```

### 动态切换存储后端

```python
# 切换到Prometheus存储
monitor.set_config({'storage': ['prometheus']})

# 启用多个存储后端
monitor.set_config({'storage': ['prometheus', 'influxdb', 'file']})

# 添加新的存储配置
monitor.update_config('prometheus', {
    'endpoint': 'http://new-prometheus:9090',
    'push_interval': 30
})
```

## 监控指标

### 系统指标

| 指标名称 | 描述 | 单位 |
|---------|------|------|
| system.cpu.usage | CPU使用率 | 百分比 |
| system.memory.usage | 内存使用率 | 百分比 |
| system.disk.read_bytes | 磁盘读取字节数 | 字节 |
| system.disk.write_bytes | 磁盘写入字节数 | 字节 |
| system.network.bytes_sent | 网络发送字节数 | 字节 |
| system.network.bytes_recv | 网络接收字节数 | 字节 |

### 业务指标

#### 交易指标

| 指标名称 | 描述 | 单位 |
|---------|------|------|
| trading.orders.total | 订单总数 | 个 |
| trading.latency.avg | 平均交易延迟 | 毫秒 |
| trading.success_rate | 交易成功率 | 百分比 |

#### 风控指标

| 指标名称 | 描述 | 单位 |
|---------|------|------|
| risk.checks.total | 风控检查总数 | 个 |
| risk.rejection_rate | 风控拒绝率 | 百分比 |
| risk.latency.avg | 平均风控延迟 | 毫秒 |

#### 数据指标

| 指标名称 | 描述 | 单位 |
|---------|------|------|
| data.load.total_records | 数据加载记录数 | 条 |
| data.processing.latency | 数据处理延迟 | 毫秒 |
| data.quality.score | 数据质量评分 | 分数 |

#### 模型指标

| 指标名称 | 描述 | 单位 |
|---------|------|------|
| model.predictions.total | 模型预测总数 | 个 |
| model.latency.avg | 平均预测延迟 | 毫秒 |
| model.accuracy | 模型准确率 | 百分比 |

## 配置最佳实践

### 1. 采集间隔设置

```python
# 开发环境：快速响应
dev_config = {'interval': 5}

# 生产环境：平衡性能和资源
prod_config = {'interval': 15}

# 高负载环境：降低采集频率
high_load_config = {'interval': 30}
```

### 2. 存储后端选择

```python
# 开发环境：文件存储
dev_storage = ['file']

# 生产环境：Prometheus + InfluxDB
prod_storage = ['prometheus', 'influxdb']

# 测试环境：仅Prometheus
test_storage = ['prometheus']
```

### 3. 监控指标优化

```python
# 启用所有指标收集
monitor.collect_business_metrics()

# 自定义指标收集
monitor.track_service_metrics('custom_service', {
    'custom_metric': 42.0,
    'response_time': 15.3
})
```

## 告警配置

### 创建告警规则

```python
# CPU使用率告警
monitor.create_alert_rule(
    condition="system.cpu.usage > 80",
    action="send_email:admin@company.com"
)

# 交易延迟告警
monitor.create_alert_rule(
    condition="trading.latency.avg > 100",
    action="send_slack:#trading-alerts"
)

# 风控拒绝率告警
monitor.create_alert_rule(
    condition="risk.rejection_rate > 5",
    action="send_sms:+1234567890"
)
```

## 性能优化建议

### 1. 指标缓存

```python
# 启用指标缓存
monitor.set_config({
    'cache_enabled': True,
    'cache_ttl': 300  # 5分钟缓存
})
```

### 2. 批量处理

```python
# 批量存储指标
monitor.set_config({
    'batch_size': 100,
    'batch_timeout': 10
})
```

### 3. 异步处理

```python
# 启用异步指标处理
monitor.set_config({
    'async_processing': True,
    'worker_threads': 4
})
```

## 故障排查

### 常见问题

1. **指标收集失败**
   ```python
   # 检查配置
   status = monitor.check_health_status()
   print(f"监控状态: {status}")
   ```

2. **存储后端连接失败**
   ```python
   # 切换到文件存储
   monitor.set_config({'storage': ['file']})
   ```

3. **性能影响**
   ```python
   # 降低采集频率
   monitor.update_config('interval', 30)
   ```

### 调试模式

```python
# 启用详细日志
monitor.set_config({
    'debug_mode': True,
    'log_level': 'DEBUG'
})
```

## 监控面板配置

### Grafana仪表板

```json
{
  "dashboard": {
    "title": "RQA2025系统监控",
    "panels": [
      {
        "title": "系统资源",
        "type": "graph",
        "targets": [
          {"expr": "system_cpu_usage_percent"},
          {"expr": "system_memory_usage_percent"}
        ]
      },
      {
        "title": "业务指标",
        "type": "graph", 
        "targets": [
          {"expr": "trading_success_rate"},
          {"expr": "model_accuracy"}
        ]
      }
    ]
  }
}
```

## 总结

通过合理的监控配置，可以：

1. **实时监控**：及时发现系统异常
2. **性能优化**：基于指标数据优化系统性能
3. **业务洞察**：通过业务指标了解系统运行状况
4. **故障预警**：通过告警机制提前发现问题

建议根据实际环境需求调整配置参数，确保监控系统既能提供足够的监控覆盖，又不会对系统性能造成显著影响。 