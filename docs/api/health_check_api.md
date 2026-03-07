# 健康检查模块 API 文档

## 概述

健康检查模块是RQA2025系统的核心监控组件，提供统一的健康检查接口、性能监控、告警管理和缓存优化功能。该模块遵循基础设施层架构设计，支持多种健康检查策略和监控集成。

## 版本信息

- **版本**: 2.1.0
- **作者**: RQA2025 Infrastructure Team
- **最后更新**: 2025-01-XX

## 核心组件

### 1. 增强健康检查器 (EnhancedHealthChecker)

主要的健康检查实现类，集成缓存、监控和告警功能。

#### 主要方法

##### `__init__(config: Optional[Dict[str, Any]] = None)`

初始化增强健康检查器。

**参数:**
- `config`: 配置字典，包含监控、告警和缓存配置

**配置选项:**
```python
{
    'monitoring_enabled': True,      # 是否启用监控
    'alerting_enabled': True,        # 是否启用告警
    'cache_enabled': True,           # 是否启用缓存
    'cache_ttl': 300,               # 缓存生存时间（秒）
    'system_metrics_cache_ttl': 60  # 系统指标缓存时间（秒）
}
```

##### `register_health_check(name: str, check_func: Callable) -> None`

注册健康检查函数。

**参数:**
- `name`: 检查名称
- `check_func`: 检查函数，应返回字典格式的检查结果

**示例:**
```python
async def custom_health_check():
    return {
        'status': 'healthy',
        'details': {'custom_metric': 'value'}
    }

checker.register_health_check('custom_check', custom_health_check)
```

##### `async perform_health_check(service: str, check_type: str, use_cache: bool = True) -> HealthCheckResult`

执行健康检查。

**参数:**
- `service`: 服务名称
- `check_type`: 检查类型
- `use_cache`: 是否使用缓存

**返回:**
- `HealthCheckResult`: 健康检查结果对象

##### `async get_comprehensive_health_status() -> Dict[str, Any]`

获取综合健康状态。

**返回:**
- 包含所有健康检查结果的字典

### 2. 缓存管理器 (HealthCheckCacheManager)

提供智能缓存策略，支持多种缓存算法和预加载功能。

#### 主要方法

##### `__init__(default_ttl: int = 300, max_size: int = 1000, policy: CachePolicy = CachePolicy.LRU)`

初始化缓存管理器。

**参数:**
- `default_ttl`: 默认缓存生存时间（秒）
- `max_size`: 最大缓存条目数
- `policy`: 缓存策略

**支持的缓存策略:**
- `CachePolicy.LRU`: 最近最少使用
- `CachePolicy.LFU`: 最少频率使用
- `CachePolicy.FIFO`: 先进先出
- `CachePolicy.PRIORITY`: 优先级

##### `get(key: str, default: Any = None) -> Any`

获取缓存值。

##### `set(key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None, priority: int = 0) -> None`

设置缓存值。

**参数:**
- `priority`: 优先级（数字越大优先级越高）

##### `get_or_compute(key: str, compute_func: Callable, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None, priority: int = 0) -> Any`

获取缓存值，如果不存在则计算并缓存。

##### `set_preload_keys(keys: List[str]) -> None`

设置预加载键列表。

##### `preload_cache(compute_funcs: Dict[str, Callable]) -> None`

预加载缓存。

##### `get_stats() -> Dict[str, Any]`

获取缓存统计信息。

**返回示例:**
```python
{
    'total_entries': 150,
    'expired_entries': 5,
    'hits': 1250,
    'misses': 50,
    'evictions': 10,
    'total_requests': 1300,
    'hit_rate_percent': 96.15,
    'max_size': 1000,
    'policy': 'lru'
}
```

### 3. Prometheus导出器 (HealthCheckPrometheusExporter)

将健康检查指标导出为Prometheus格式，支持Grafana监控。

#### 主要方法

##### `record_health_check(service: str, check_type: str, status: str, response_time: float, error_code: Optional[str] = None, instance: str = "default") -> None`

记录健康检查指标。

##### `record_system_metrics(host: str, cpu_percent: float, memory_bytes: int, disk_usage: Dict[str, float], instance: str = "default") -> None`

记录系统指标。

##### `record_cache_metrics(cache_type: str, hit_rate: float, total_entries: int, evictions: int, policy: str = "lru", instance: str = "default") -> None`

记录缓存指标。

##### `export_grafana_dashboard(filepath: Optional[str] = None) -> str`

导出Grafana仪表板配置。

### 4. 告警管理器 (AlertManager)

实现基于性能阈值的自动告警机制。

#### 主要方法

##### `add_alert_rule(rule: AlertRule) -> None`

添加告警规则。

**告警规则示例:**
```python
rule = AlertRule(
    name="high_cpu_usage",
    description="CPU使用率过高",
    metric_name="rqa_system_cpu_usage_percent",
    threshold=80.0,
    comparison=">",
    severity=AlertSeverity.WARNING,
    duration=300,
    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
    escalation_delay=600
)
```

##### `check_alert_condition(metric_name: str, metric_value: Union[float, int], labels: Optional[Dict[str, str]] = None) -> List[Alert]`

检查告警条件。

##### `acknowledge_alert(alert_id: str, user: str, notes: Optional[str] = None) -> bool`

确认告警。

##### `resolve_alert(alert_id: str, notes: Optional[str] = None) -> bool`

解决告警。

## API端点

### 基础健康检查

#### `GET /health/`

基础健康检查端点。

**响应示例:**
```json
{
    "timestamp": "2025-01-XXTXX:XX:XX",
    "status": "healthy",
    "services": {
        "system": "healthy",
        "database": "healthy",
        "cache": "healthy"
    },
    "overall_health": "healthy"
}
```

#### `GET /health/ready`

就绪检查端点。

**响应示例:**
```json
{
    "timestamp": "2025-01-XXTXX:XX:XX",
    "ready": true,
    "checks": [
        {
            "type": "system_resources",
            "status": "healthy",
            "response_time": 0.05
        }
    ]
}
```

#### `GET /health/live`

存活检查端点。

### 监控指标

#### `GET /health/metrics`

Prometheus指标端点。

#### `GET /health/metrics/summary`

指标摘要端点。

### 缓存管理

#### `GET /health/cache/stats`

缓存统计信息。

#### `POST /health/cache/clear`

清除缓存。

**参数:**
- `pattern`: 清除模式，支持通配符

### 告警管理

#### `GET /health/alerts`

获取告警列表。

**查询参数:**
- `status`: 告警状态过滤
- `severity`: 告警严重程度过滤
- `limit`: 返回告警数量限制

#### `POST /health/alerts/{alert_id}/acknowledge`

确认告警。

**参数:**
- `user`: 确认用户
- `notes`: 确认备注

#### `POST /health/alerts/{alert_id}/resolve`

解决告警。

**参数:**
- `notes`: 解决备注

#### `GET /health/alerts/summary`

告警摘要。

### 系统状态

#### `GET /health/check/{check_type}`

特定健康检查。

**参数:**
- `service`: 服务名称
- `use_cache`: 是否使用缓存

#### `GET /health/system/metrics`

系统指标。

#### `GET /health/config`

配置信息。

#### `GET /health/status`

详细状态信息。

## 使用示例

### 基本使用

```python
from src.infrastructure.health import get_enhanced_health_checker

# 获取健康检查器实例
checker = get_enhanced_health_checker()

# 执行健康检查
result = await checker.perform_health_check('my_service', 'availability')

# 获取综合健康状态
status = await checker.get_comprehensive_health_status()
```

### 缓存使用

```python
from src.infrastructure.health import get_cache_manager

# 获取缓存管理器
cache_manager = get_cache_manager()

# 设置缓存
cache_manager.set('key', 'value', ttl=600, priority=10)

# 获取或计算缓存
value = cache_manager.get_or_compute('key', compute_function, ttl=300)

# 预加载缓存
cache_manager.set_preload_keys(['frequently_used_key'])
cache_manager.preload_cache({'frequently_used_key': compute_function})
```

### 告警配置

```python
from src.infrastructure.health import get_alert_manager, AlertRule, AlertSeverity, NotificationChannel

# 获取告警管理器
alert_manager = get_alert_manager()

# 创建告警规则
rule = AlertRule(
    name="custom_alert",
    description="自定义告警",
    metric_name="custom_metric",
    threshold=100,
    comparison=">",
    severity=AlertSeverity.CRITICAL,
    duration=60,
    notification_channels=[NotificationChannel.EMAIL]
)

# 添加告警规则
alert_manager.add_alert_rule(rule)

# 检查告警条件
alerts = alert_manager.check_alert_condition('custom_metric', 150)
```

### Prometheus集成

```python
from src.infrastructure.health import get_prometheus_exporter

# 获取Prometheus导出器
exporter = get_prometheus_exporter()

# 记录健康检查指标
exporter.record_health_check(
    service='my_service',
    check_type='availability',
    status='healthy',
    response_time=0.05
)

# 记录系统指标
exporter.record_system_metrics(
    host='server-01',
    cpu_percent=45.2,
    memory_bytes=8589934592,
    disk_usage={'/': 75.5, '/data': 60.2}
)

# 导出Grafana仪表板
dashboard_json = exporter.export_grafana_dashboard()
```

## 配置说明

### 环境变量

```bash
# 健康检查配置
HEALTH_CHECK_CACHE_TTL=300
HEALTH_CHECK_MAX_CACHE_SIZE=1000
HEALTH_CHECK_CACHE_POLICY=LRU

# 监控配置
HEALTH_CHECK_MONITORING_ENABLED=true
HEALTH_CHECK_ALERTING_ENABLED=true

# Prometheus配置
PROMETHEUS_METRICS_PORT=9090
PROMETHEUS_METRICS_PATH=/metrics
```

### 配置文件

```yaml
health_check:
  cache:
    default_ttl: 300
    max_size: 1000
    policy: LRU
  
  monitoring:
    enabled: true
    metrics_export_interval: 30
  
  alerting:
    enabled: true
    default_escalation_delay: 300
  
  notifications:
    email:
      enabled: true
      smtp_server: smtp.example.com
      smtp_port: 587
      username: alerts@example.com
      password: ${EMAIL_PASSWORD}
    
    webhook:
      enabled: true
      url: https://webhook.example.com/alerts
      timeout: 30
    
    slack:
      enabled: true
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: #alerts
```

## 性能优化建议

### 1. 缓存策略选择

- **LRU**: 适用于访问模式相对均匀的场景
- **LFU**: 适用于某些键被频繁访问的场景
- **PRIORITY**: 适用于需要保证重要数据不被驱逐的场景

### 2. 预加载配置

```python
# 设置预加载键
cache_manager.set_preload_keys([
    'system_metrics',
    'service_dependencies',
    'configuration'
])

# 定期预加载
import asyncio

async def preload_worker():
    while True:
        cache_manager.preload_cache({
            'system_metrics': get_system_metrics,
            'service_dependencies': get_service_dependencies,
            'configuration': get_configuration
        })
        await asyncio.sleep(60)

asyncio.create_task(preload_worker())
```

### 3. 告警规则优化

- 合理设置阈值和持续时间
- 使用标签进行分组和过滤
- 配置适当的升级延迟

## 故障排除

### 常见问题

#### 1. 缓存命中率低

**可能原因:**
- 缓存TTL设置过短
- 缓存大小限制过小
- 缓存策略不适合访问模式

**解决方案:**
- 调整缓存TTL
- 增加缓存大小
- 选择合适的缓存策略

#### 2. 告警频繁触发

**可能原因:**
- 阈值设置过低
- 持续时间设置过短
- 缺少抑制规则

**解决方案:**
- 调整告警阈值
- 增加持续时间要求
- 配置抑制规则

#### 3. 监控指标缺失

**可能原因:**
- Prometheus客户端未安装
- 指标记录函数调用失败
- 标签配置错误

**解决方案:**
- 安装prometheus-client
- 检查指标记录调用
- 验证标签配置

### 日志分析

健康检查模块使用结构化日志，可以通过以下方式分析：

```python
import logging

# 设置日志级别
logging.getLogger('src.infrastructure.health').setLevel(logging.DEBUG)

# 查看缓存统计
cache_stats = cache_manager.get_stats()
print(f"Cache hit rate: {cache_stats['hit_rate_percent']}%")

# 查看告警摘要
alert_summary = alert_manager.get_alert_summary()
print(f"Active alerts: {alert_summary['active_count']}")
```

## 扩展开发

### 自定义健康检查

```python
class CustomHealthChecker:
    async def check_database_connection(self):
        try:
            # 执行数据库连接检查
            result = await self.db.ping()
            return {
                'status': 'healthy' if result else 'unhealthy',
                'details': {'connection_time': 0.05}
            }
        except Exception as e:
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }

# 注册自定义检查
checker = get_enhanced_health_checker()
checker.register_health_check('database_connection', CustomHealthChecker().check_database_connection)
```

### 自定义通知渠道

```python
class CustomNotificationChannel:
    def __init__(self, config):
        self.config = config
    
    async def send_notification(self, alert):
        # 实现自定义通知逻辑
        pass

# 注册自定义通知渠道
alert_manager = get_alert_manager()
alert_manager.register_notification_channel('custom', CustomNotificationChannel(config))
```

## 更新日志

### v2.1.0 (2025-01-XX)
- 新增智能缓存策略支持
- 增强Prometheus指标导出
- 完善告警管理功能
- 添加Grafana仪表板配置
- 优化性能和稳定性

### v2.0.0 (2025-01-XX)
- 重构健康检查架构
- 统一接口设计
- 集成缓存和监控功能

### v1.0.0 (2025-01-XX)
- 初始版本发布
- 基础健康检查功能
- 简单监控支持
