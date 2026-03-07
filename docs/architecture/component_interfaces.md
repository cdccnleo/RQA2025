# 基础设施层监控组件接口文档

## 概述

本文档描述了基础设施层监控模块中各组件间的接口规范和协作机制。基于组件化架构设计，各组件通过定义明确的接口进行协作。

## 架构概览

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Monitoring      │    │ StatsCollector   │    │ AlertManager    │
│ Coordinator     │◄──►│                  │◄──►│                 │
│                 │    │ - collect_stats()│    │ - check_alerts()│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ MetricsExporter │    │ DataPersistor    │    │ NotificationMgr │
│                 │    │                  │    │                 │
│ - export()      │    │ - persist()      │    │ - send()        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 核心组件接口

### 1. MonitoringCoordinator (监控协调器)

**职责**: 协调监控系统的各个组件，提供统一的监控生命周期管理。

#### 接口定义

```python
class MonitoringCoordinator:
    def __init__(self, pool_name: str, config: MonitoringConfig) -> None:
        """初始化监控协调器"""

    def set_components(self, stats_collector, alert_manager, metrics_exporter) -> None:
        """设置监控组件"""

    def start_monitoring(self) -> bool:
        """启动监控"""

    def stop_monitoring(self) -> bool:
        """停止监控"""

    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""

    def _execute_monitoring_cycle(self) -> None:
        """执行监控周期（内部方法）"""
```

#### 依赖关系
- **依赖**: StatsCollector, AlertManager, MetricsExporter
- **被依赖**: LoggerPoolMonitorRefactored

### 2. StatsCollector (统计收集器)

**职责**: 收集和计算各种统计信息，支持历史数据管理和趋势分析。

#### 接口定义

```python
class StatsCollector:
    def __init__(self, pool_name: str, config: LoggerPoolStatsConfig) -> None:
        """初始化统计收集器"""

    def collect_stats(self) -> Optional[Dict[str, Any]]:
        """收集统计信息"""

    def get_current_stats(self) -> Optional[Dict[str, Any]]:
        """获取当前统计信息"""

    def get_history_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取历史统计信息"""

    def get_access_times(self, limit: int = 100) -> List[float]:
        """获取访问时间记录"""

    def record_access_time(self, access_time: float) -> None:
        """记录访问时间"""

    def calculate_percentiles(self, data: List[float], percentiles: List[float]) -> Dict[str, float]:
        """计算百分位数"""

    def analyze_trends(self, metric_name: str, window_size: int = 10) -> Dict[str, Any]:
        """分析趋势"""
```

#### 数据格式规范

**统计数据格式**:
```json
{
  "pool_name": "logger_pool_1",
  "pool_size": 45,
  "max_size": 100,
  "created_count": 1250,
  "hit_count": 10000,
  "hit_rate": 0.85,
  "memory_usage_mb": 128.5,
  "avg_access_time": 0.003,
  "timestamp": "2025-10-27T10:30:00Z"
}
```

### 3. AlertManager (告警管理器)

**职责**: 管理和执行告警规则，支持多种告警条件和通知渠道。

#### 接口定义

```python
class AlertManager:
    def __init__(self, pool_name: str, alert_thresholds: Dict[str, float]) -> None:
        """初始化告警管理器"""

    def add_alert_rule(self, rule: AlertRuleConfig) -> bool:
        """添加告警规则"""

    def remove_alert_rule(self, rule_id: str) -> bool:
        """移除告警规则"""

    def get_alert_rule(self, rule_id: str) -> Optional[AlertRuleConfig]:
        """获取告警规则"""

    def get_all_alert_rules(self) -> List[AlertRuleConfig]:
        """获取所有告警规则"""

    def process_alerts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理告警"""

    def get_alert_history(self, limit: int = 100, level: Optional[str] = None,
                         status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取告警历史"""

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""

    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计信息"""
```

#### 告警数据格式

**告警信息格式**:
```json
{
  "alert_id": "alert_001_1635323400",
  "rule_id": "high_cpu",
  "rule_name": "CPU使用率过高",
  "description": "CPU使用率超过80%",
  "level": "warning",
  "status": "active",
  "triggered_at": "2025-10-27T10:30:00Z",
  "data": {
    "cpu_percent": 85.5,
    "memory_percent": 70.2
  },
  "channels": ["console", "email"],
  "acknowledged": false
}
```

### 4. MetricsExporter (指标导出器)

**职责**: 将监控指标导出为各种格式，支持Prometheus、JSON等多种格式。

#### 接口定义

```python
class MetricsExporter:
    def __init__(self, pool_name: str, config: PrometheusExportConfig) -> None:
        """初始化指标导出器"""

    def export_metrics(self, stats: Dict[str, Any]) -> bool:
        """导出指标"""

    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式的指标"""

    def get_json_metrics(self) -> str:
        """获取JSON格式的指标"""

    def get_export_status(self) -> Dict[str, Any]:
        """获取导出状态"""

    def clear_cache(self) -> None:
        """清空导出缓存"""

    def export_to_file(self, format_type: str = 'prometheus',
                      file_path: Optional[str] = None) -> bool:
        """导出到文件"""

    def get_supported_formats(self) -> list:
        """获取支持的导出格式"""

    def validate_export_data(self, format_type: str) -> bool:
        """验证导出数据"""
```

#### Prometheus格式规范

```
# HELP logger_pool_size Logger pool current size
# TYPE logger_pool_size gauge
logger_pool_size{pool="logger_pool_1"} 45

# HELP logger_pool_hit_rate Cache hit rate
# TYPE logger_pool_hit_rate gauge
logger_pool_hit_rate{pool="logger_pool_1"} 0.85
```

### 5. DataPersistor (数据持久化器)

**职责**: 负责监控数据的存储、检索和管理，支持文件和数据库存储。

#### 接口定义

```python
class DataPersistor:
    def __init__(self, pool_name: str, config: DataPersistenceConfig) -> None:
        """初始化数据持久化器"""

    def persist_data(self, data: Dict[str, Any]) -> bool:
        """持久化数据"""

    def retrieve_data(self, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """检索数据"""

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """清理旧数据"""

    def export_data(self, file_path: str, format_type: str = 'json') -> bool:
        """导出数据"""
```

## 参数对象规范

### MonitoringConfig (监控配置)

```python
@dataclass
class MonitoringConfig:
    collection_interval: int = 60  # 收集间隔(秒)
    max_history_size: int = 1000   # 最大历史记录数
    alert_thresholds: Dict[str, float] = field(default_factory=dict)  # 告警阈值
    enable_prometheus_export: bool = True  # 启用Prometheus导出
    prometheus_port: int = 9090  # Prometheus端口
```

### AlertRuleConfig (告警规则配置)

```python
@dataclass
class AlertRuleConfig:
    rule_id: str  # 规则ID
    name: str     # 规则名称
    description: str  # 规则描述
    condition: AlertConditionConfig  # 触发条件
    level: str = "warning"  # 告警级别
    channels: List[str] = field(default_factory=lambda: ["console"])  # 通知渠道
    enabled: bool = True  # 是否启用
    cooldown: int = 300   # 冷却时间(秒)
    metadata: Optional[Dict[str, Any]] = field(default=None)  # 扩展元数据
```

### AlertConditionConfig (告警条件配置)

```python
@dataclass
class AlertConditionConfig:
    field: str  # 监控字段
    value: Any  # 比较值
    operator: str = "gt"  # 操作符: gt, lt, eq, ne, ge, le
    threshold: Optional[float] = field(default=None)  # 阈值
    comparison_type: str = "absolute"  # 比较类型: absolute, percentage
```

## 组件协作协议

### 1. 初始化协议

```python
# 1. 创建配置对象
monitoring_config = MonitoringConfig(collection_interval=30)
stats_config = LoggerPoolStatsConfig()
export_config = PrometheusExportConfig()

# 2. 创建组件实例
coordinator = MonitoringCoordinator("pool_1", monitoring_config)
stats_collector = StatsCollector("pool_1", stats_config)
alert_manager = AlertManager("pool_1", monitoring_config.alert_thresholds)
metrics_exporter = MetricsExporter("pool_1", export_config)

# 3. 设置组件协作关系
coordinator.set_components(stats_collector, alert_manager, metrics_exporter)
```

### 2. 运行时协议

```python
# 启动监控
coordinator.start_monitoring()

# 监控循环会自动执行：
# 1. stats_collector.collect_stats()
# 2. alert_manager.check_alerts(stats)
# 3. metrics_exporter.export_metrics(stats)

# 获取状态
status = coordinator.get_monitoring_status()

# 停止监控
coordinator.stop_monitoring()
```

### 3. 数据流协议

```
监控数据流:
StatsCollector → AlertManager → MetricsExporter
     ↓              ↓              ↓
  收集统计      检查告警      导出指标
     ↓              ↓              ↓
DataPersistor ← AlertHistory ← Prometheus/JSON
  数据存储      历史记录      格式导出
```

## 异常处理协议

### 统一异常处理

```python
from src.infrastructure.monitoring.core.unified_exception_handler import (
    handle_monitoring_exception,
    MonitoringException,
    ValidationError
)

# 使用装饰器
@handle_monitoring_exception("data_collection")
def collect_data():
    # 业务逻辑
    pass

# 手动处理异常
try:
    result = risky_operation()
except Exception as e:
    result = handle_exception(e, "risky_operation", strategy="retry")
```

### 异常层次结构

```
MonitoringException (基础异常)
├── ValidationError (数据验证异常)
├── ConfigurationError (配置异常)
├── ConnectionError (连接异常)
├── DataPersistenceError (数据持久化异常)
├── AlertProcessingError (告警处理异常)
└── NotificationError (通知异常)
```

## 扩展机制

### 1. 自定义告警规则

```python
# 创建自定义告警规则
custom_rule = AlertRuleConfig(
    rule_id="custom_memory",
    name="自定义内存告警",
    description="内存使用异常检测",
    condition=AlertConditionConfig(
        field="memory_mb",
        value=500,
        operator="gt"
    ),
    level="critical",
    channels=["email", "slack"]
)

alert_manager.add_alert_rule(custom_rule)
```

### 2. 自定义指标导出器

```python
class CustomMetricsExporter(MetricsExporter):
    def export_metrics(self, stats: Dict[str, Any]) -> bool:
        # 自定义导出逻辑
        return super().export_metrics(stats)
```

### 3. 插件化扩展

```python
# 注册自定义异常处理策略
handler.add_strategy('custom_retry', CustomRetryStrategy())

# 使用自定义策略
@handle_monitoring_exception("custom_operation", strategy="custom_retry")
def custom_operation():
    pass
```

## 性能优化建议

### 1. 异步处理

```python
import asyncio

async def collect_stats_async(self) -> Dict[str, Any]:
    """异步收集统计信息"""
    # 实现异步收集逻辑
    pass
```

### 2. 缓存优化

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_stats(self, key: str) -> Dict[str, Any]:
    """缓存统计信息"""
    return self._calculate_stats(key)
```

### 3. 批量处理

```python
def batch_export_metrics(self, stats_list: List[Dict[str, Any]]) -> bool:
    """批量导出指标"""
    # 实现批量处理逻辑
    pass
```

## 测试接口

### 单元测试接口

```python
# 测试StatsCollector
stats_collector = StatsCollector("test_pool", config)
stats = stats_collector.collect_stats()
assert stats is not None

# 测试AlertManager
alert_manager = AlertManager("test_pool", thresholds)
alerts = alert_manager.process_alerts(test_data)
assert len(alerts) == expected_count
```

### 集成测试接口

```python
# 测试完整协作流程
monitor = LoggerPoolMonitorRefactored("test_pool", monitoring_config, ...)
with monitor:
    # 执行测试操作
    stats = monitor.collect_current_stats()
    alerts = monitor.get_active_alerts()
    metrics = monitor.get_prometheus_metrics()

    # 验证集成结果
    assert stats is not None
    assert isinstance(alerts, list)
    assert len(metrics) > 0
```

## 版本兼容性

### API版本控制

- **v1.0**: 基础组件接口
- **v1.1**: 添加异步支持
- **v1.2**: 添加插件化扩展

### 向后兼容性保证

- 保持现有接口不变
- 新功能通过可选参数添加
- 废弃功能提前公告并提供迁移路径

---

**文档版本**: v1.0
**更新日期**: 2025年10月27日
**维护者**: AI Assistant
