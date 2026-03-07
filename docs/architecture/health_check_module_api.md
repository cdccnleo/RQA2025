# RQA2025 健康检查模块 API 文档

## 概述

健康检查模块是RQA2025基础设施层的核心组件，提供统一的健康检查接口、性能监控、告警管理和监控集成功能。

## 版本信息

- **版本**: 2.1.0
- **最后更新**: 2025-01-27
- **维护团队**: RQA2025 Infrastructure Team

## 核心组件

### 1. 健康检查器 (Health Checker)

#### 基础健康检查器 (BasicHealthChecker)

```python
from src.infrastructure.health import BasicHealthChecker

# 创建基础健康检查器
checker = BasicHealthChecker()

# 执行健康检查
result = await checker.check_health()
```

**主要方法**:
- `check_health()`: 执行基础健康检查
- `check_readiness()`: 检查服务就绪状态
- `add_dependency_check()`: 添加依赖服务检查

#### 增强健康检查器 (EnhancedHealthChecker)

```python
from src.infrastructure.health import EnhancedHealthChecker

# 创建增强健康检查器
checker = EnhancedHealthChecker(
    cache_manager=cache_manager,
    prometheus_exporter=prometheus_exporter,
    alert_manager=alert_manager
)

# 执行增强健康检查
result = await checker.check_health()
```

**增强功能**:
- 性能指标收集
- 缓存集成
- Prometheus指标导出
- 告警管理

### 2. 缓存管理器 (Cache Manager)

#### 基本用法

```python
from src.infrastructure.health import get_cache_manager

# 获取缓存管理器实例
cache_manager = get_cache_manager()

# 设置缓存
cache_manager.set("health_status", health_data, ttl=300)

# 获取缓存
status = cache_manager.get("health_status")

# 智能获取或计算
status = cache_manager.get_or_compute(
    "health_status", 
    compute_health_status, 
    ttl=300
)
```

**缓存策略**:
- LRU (最近最少使用)
- LFU (最少频率使用)
- FIFO (先进先出)
- Priority (优先级)

#### 高级功能

```python
# 设置预加载键
cache_manager.set_preload_keys(["health_status", "system_metrics"])

# 预加载缓存
compute_funcs = {
    "health_status": compute_health_status,
    "system_metrics": compute_system_metrics
}
cache_manager.preload_cache(compute_funcs)

# 获取缓存统计
stats = cache_manager.get_stats()
```

### 3. Prometheus导出器 (Prometheus Exporter)

#### 基本配置

```python
from src.infrastructure.health import get_prometheus_exporter

# 获取Prometheus导出器
exporter = get_prometheus_exporter()

# 记录健康检查指标
exporter.record_health_check(
    service="api",
    check_type="liveness",
    status="healthy",
    response_time=0.1
)

# 记录系统指标
exporter.record_system_metrics(
    host="server-01",
    cpu_percent=45.2,
    memory_bytes=8589934592,
    disk_usage={"root": 0.65}
)
```

#### 指标类型

- **健康状态指标**: `rqa_health_status`
- **响应时间指标**: `rqa_health_response_time_seconds`
- **系统资源指标**: `rqa_system_cpu_percent`, `rqa_system_memory_percent`
- **缓存性能指标**: `rqa_cache_hit_rate`
- **告警统计指标**: `rqa_alert_count`

### 4. 告警管理器 (Alert Manager)

#### 基本用法

```python
from src.infrastructure.health import get_alert_manager

# 获取告警管理器
alert_manager = get_alert_manager()

# 发送告警
alert_manager.send_alert(
    severity="critical",
    message="CPU使用率过高",
    labels={"service": "api", "instance": "server-01"},
    annotations={"description": "CPU使用率超过90%"}
)

# 获取活跃告警
active_alerts = alert_manager.get_active_alerts()

# 确认告警
alert_manager.acknowledge_alert(alert_id, user="admin")
```

#### 告警严重程度

- **INFO**: 信息级别
- **WARNING**: 警告级别
- **CRITICAL**: 严重级别
- **EMERGENCY**: 紧急级别

### 5. 性能优化器 (Performance Optimizer)

#### 基本用法

```python
from src.infrastructure.health import get_performance_optimizer

# 获取性能优化器
optimizer = get_performance_optimizer(
    cache_manager=cache_manager,
    prometheus_exporter=prometheus_exporter
)

# 记录性能指标
optimizer.record_metric("response_time", 0.15)
optimizer.record_metric("cache_hit_rate", 0.85)

# 获取性能报告
report = optimizer.get_performance_report()

# 获取优化建议
suggestions = optimizer.analyze_performance()
```

#### 自动优化功能

- 智能缓存策略调整
- 自适应TTL管理
- 性能趋势分析
- 自动阈值调整

### 6. Grafana集成 (Grafana Integration)

#### 基本配置

```python
from src.infrastructure.health import get_grafana_integration

# 创建Grafana集成
grafana = get_grafana_integration(
    grafana_url="http://grafana:3000",
    api_key="your-api-key",
    org_id=1
)

# 部署监控仪表板
results = grafana.deploy_all_dashboards()

# 导出仪表板配置
grafana.export_dashboard_config("dashboards.json")
```

#### 预定义仪表板

1. **健康检查监控仪表板**
   - 系统健康状态概览
   - CPU和内存使用率
   - 健康检查响应时间
   - 缓存命中率
   - 告警统计

2. **性能监控仪表板**
   - 关键性能指标趋势
   - 性能优化建议

### 7. 告警规则引擎 (Alert Rule Engine)

#### 规则定义

```python
from src.infrastructure.health import AlertRule, AlertSeverity

# 创建告警规则
rule = AlertRule(
    name="high_cpu_usage",
    description="CPU使用率过高告警",
    query="rqa_system_cpu_percent > 80",
    severity=AlertSeverity.CRITICAL,
    threshold=80.0,
    duration="5m",
    auto_threshold=True,
    suppress_conditions=["time: 23:00-06:00"]
)

# 添加规则到引擎
rule_engine = get_alert_rule_engine()
rule_engine.add_rule(rule)
```

#### 规则管理

```python
# 更新规则
rule_engine.update_rule("high_cpu_usage", {
    "threshold": 85.0,
    "duration": "3m"
})

# 抑制告警
rule_engine.suppress_alert("high_cpu_usage", "2h", "维护期间")

# 获取规则统计
stats = rule_engine.get_rule_statistics()
```

## 配置示例

### 1. 基础配置

```python
# config/health_check.yaml
health_check:
  enabled: true
  interval: 30s
  timeout: 10s
  
  cache:
    enabled: true
    ttl: 300s
    max_size: 1000
    policy: "lru"
  
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"
  
  alerting:
    enabled: true
    rules_file: "alert_rules.yaml"
    notification_channels:
      - email: "admin@example.com"
      - slack: "#alerts"
```

### 2. 告警规则配置

```yaml
# config/alert_rules.yaml
groups:
  - name: system_alerts
    rules:
      - alert: HighCPUUsage
        expr: rqa_system_cpu_percent > 80
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "CPU使用率过高"
          description: "CPU使用率超过80%持续5分钟"
      
      - alert: HighMemoryUsage
        expr: rqa_system_memory_percent > 85
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "内存使用率过高"
          description: "内存使用率超过85%持续3分钟"
```

### 3. Grafana配置

```yaml
# config/grafana.yaml
grafana:
  url: "http://grafana:3000"
  api_key: "your-api-key"
  org_id: 1
  
  dashboards:
    health_monitoring:
      enabled: true
      refresh: "30s"
    performance_monitoring:
      enabled: true
      refresh: "1m"
```

## 使用示例

### 1. 完整集成示例

```python
#!/usr/bin/env python3
"""
健康检查模块完整集成示例
"""

import asyncio
from src.infrastructure.health import (
    get_enhanced_health_checker,
    get_cache_manager,
    get_prometheus_exporter,
    get_alert_manager,
    get_performance_optimizer,
    get_grafana_integration
)

async def main():
    """主函数"""
    # 初始化组件
    cache_manager = get_cache_manager()
    prometheus_exporter = get_prometheus_exporter()
    alert_manager = get_alert_manager()
    
    # 创建增强健康检查器
    health_checker = get_enhanced_health_checker(
        cache_manager=cache_manager,
        prometheus_exporter=prometheus_exporter,
        alert_manager=alert_manager
    )
    
    # 创建性能优化器
    optimizer = get_performance_optimizer(
        cache_manager=cache_manager,
        prometheus_exporter=prometheus_exporter
    )
    
    # 执行健康检查
    health_result = await health_checker.check_health()
    print(f"健康检查结果: {health_result}")
    
    # 获取性能报告
    performance_report = optimizer.get_performance_report()
    print(f"性能报告: {performance_report}")
    
    # 部署Grafana仪表板（如果配置了）
    try:
        grafana = get_grafana_integration(
            grafana_url="http://localhost:3000",
            api_key="demo-key"
        )
        results = grafana.deploy_all_dashboards()
        print(f"仪表板部署结果: {results}")
    except Exception as e:
        print(f"Grafana集成失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 自定义健康检查

```python
#!/usr/bin/env python3
"""
自定义健康检查示例
"""

from src.infrastructure.health import BasicHealthChecker
import asyncio

class CustomHealthChecker(BasicHealthChecker):
    """自定义健康检查器"""
    
    async def check_database_health(self):
        """检查数据库健康状态"""
        try:
            # 执行数据库连接测试
            # ... 数据库检查逻辑 ...
            return {"status": "healthy", "response_time": 0.05}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_external_service(self):
        """检查外部服务健康状态"""
        try:
            # 执行外部服务检查
            # ... 外部服务检查逻辑 ...
            return {"status": "healthy", "response_time": 0.12}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_health(self):
        """执行完整健康检查"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        # 执行基础检查
        basic_result = await super().check_health()
        results["checks"]["basic"] = basic_result
        
        # 执行自定义检查
        db_result = await self.check_database_health()
        results["checks"]["database"] = db_result
        
        external_result = await self.check_external_service()
        results["checks"]["external_service"] = external_result
        
        # 确定整体状态
        if any(check["status"] != "healthy" for check in results["checks"].values()):
            results["status"] = "degraded"
        
        return results

async def main():
    """主函数"""
    checker = CustomHealthChecker()
    result = await checker.check_health()
    print(f"自定义健康检查结果: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. 告警规则管理

```python
#!/usr/bin/env python3
"""
告警规则管理示例
"""

from src.infrastructure.health import (
    get_alert_rule_engine,
    AlertRule,
    AlertSeverity
)

def setup_alert_rules():
    """设置告警规则"""
    rule_engine = get_alert_rule_engine()
    
    # 系统资源告警规则
    rules = [
        AlertRule(
            name="high_cpu_usage",
            description="CPU使用率过高",
            query="rqa_system_cpu_percent > 80",
            severity=AlertSeverity.CRITICAL,
            threshold=80.0,
            duration="5m",
            auto_threshold=True
        ),
        AlertRule(
            name="high_memory_usage",
            description="内存使用率过高",
            query="rqa_system_memory_percent > 85",
            severity=AlertSeverity.WARNING,
            threshold=85.0,
            duration="3m"
        ),
        AlertRule(
            name="slow_response_time",
            description="响应时间过慢",
            query="rqa_health_response_time_seconds > 2",
            severity=AlertSeverity.WARNING,
            threshold=2.0,
            duration="2m"
        )
    ]
    
    # 添加规则
    for rule in rules:
        rule_engine.add_rule(rule)
        print(f"告警规则已添加: {rule.name}")
    
    return rule_engine

def main():
    """主函数"""
    # 设置告警规则
    rule_engine = setup_alert_rules()
    
    # 获取规则统计
    stats = rule_engine.get_rule_statistics()
    print(f"规则统计: {stats}")
    
    # 获取活跃告警
    active_alerts = rule_engine.get_active_alerts()
    print(f"活跃告警: {len(active_alerts)}")

if __name__ == "__main__":
    main()
```

## 监控和调试

### 1. 日志配置

```python
# 配置健康检查模块日志
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 设置健康检查模块日志级别
logging.getLogger('src.infrastructure.health').setLevel(logging.DEBUG)
```

### 2. 性能监控

```python
# 获取性能指标
from src.infrastructure.health import get_performance_optimizer

optimizer = get_performance_optimizer()
report = optimizer.get_performance_report()

print("性能指标统计:")
for metric_name, stats in report['metrics_summary'].items():
    print(f"  {metric_name}:")
    print(f"    平均值: {stats['mean']:.2f}")
    print(f"    趋势: {stats['trend']}")
    print(f"    标准差: {stats['std_dev']:.2f}")
```

### 3. 缓存性能

```python
# 获取缓存性能统计
from src.infrastructure.health import get_cache_manager

cache_manager = get_cache_manager()
stats = cache_manager.get_stats()

print(f"缓存命中率: {stats['hit_rate']:.2%}")
print(f"缓存条目数: {stats['total_entries']}")
print(f"缓存驱逐次数: {stats['evictions']}")
```

## 故障排除

### 1. 常见问题

**问题**: 健康检查超时
**解决方案**: 检查网络连接和依赖服务状态，调整超时配置

**问题**: 缓存命中率低
**解决方案**: 调整TTL设置，启用预加载，优化缓存策略

**问题**: Prometheus指标导出失败
**解决方案**: 检查Prometheus服务状态，验证网络连接

### 2. 调试技巧

```python
# 启用详细日志
import logging
logging.getLogger('src.infrastructure.health').setLevel(logging.DEBUG)

# 检查组件状态
cache_manager = get_cache_manager()
print(f"缓存管理器状态: {cache_manager.get_stats()}")

prometheus_exporter = get_prometheus_exporter()
print(f"Prometheus导出器状态: {prometheus_exporter.get_metrics_summary()}")
```

## 最佳实践

### 1. 性能优化

- 合理设置缓存TTL，避免频繁计算
- 使用预加载机制预热热点数据
- 定期分析性能指标，调整配置参数

### 2. 监控配置

- 设置合理的告警阈值，避免误报
- 使用告警抑制机制，减少告警噪音
- 定期审查和优化告警规则

### 3. 高可用性

- 配置多个健康检查端点
- 实现健康检查的负载均衡
- 设置告警升级机制

## 更新日志

### v2.1.0 (2025-01-27)
- 新增性能优化器
- 增强Grafana集成
- 新增智能告警规则引擎
- 改进缓存管理策略

### v2.0.0 (2025-01-20)
- 重构健康检查架构
- 统一接口设计
- 集成Prometheus监控
- 新增告警管理功能

### v1.0.0 (2025-01-10)
- 初始版本发布
- 基础健康检查功能
- 简单缓存机制

## 技术支持

如有问题或建议，请联系：
- **邮箱**: infrastructure@rqa2025.com
- **文档**: https://docs.rqa2025.com/health
- **问题反馈**: https://github.com/rqa2025/issues
