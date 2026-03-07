# FeaturesMonitor API 文档

## 概述
`FeaturesMonitor` 是特征层的统一监控器，提供特征层组件的统一监控接口，包括性能监控、指标收集、告警管理等功能。

## 类和方法

### FeaturesMonitor
特征层统一监控器，提供组件注册、指标收集、实时告警、性能分析等功能

#### 初始化
```python
def __init__(self, config: Optional[Dict] = None):
```

**参数**:
- `config` (dict, optional): 监控配置字典

**示例**:
```python
from src.features.monitoring import FeaturesMonitor

# 使用默认配置
monitor = FeaturesMonitor()

# 使用自定义配置
config = {
    'monitor_interval': 5.0,
    'thresholds': {
        'cpu_usage': 80.0,
        'memory_usage': 80.0,
        'error_rate': 5.0
    }
}
monitor = FeaturesMonitor(config=config)
```

#### 方法

##### register_component(name: str, component_type: str, component: Any = None)
注册组件到监控系统

**参数**:
- `name` (str): 组件名称
- `component_type` (str): 组件类型
- `component` (Any, optional): 组件实例

**返回**:
- `None`

**示例**:
```python
# 注册特征工程器
monitor.register_component(
    name="feature_engineer",
    component_type="processor",
    component=engineer
)

# 注册技术指标处理器
monitor.register_component(
    name="technical_processor",
    component_type="processor"
)
```

##### unregister_component(name: str)
从监控系统注销组件

**参数**:
- `name` (str): 组件名称

**返回**:
- `None`

**示例**:
```python
monitor.unregister_component("feature_engineer")
```

##### update_component_status(name: str, status: str, metrics: Optional[Dict] = None)
更新组件状态和指标

**参数**:
- `name` (str): 组件名称
- `status` (str): 组件状态（"running", "stopped", "error"等）
- `metrics` (dict, optional): 组件指标

**返回**:
- `None`

**示例**:
```python
# 更新组件状态
monitor.update_component_status(
    name="feature_engineer",
    status="running",
    metrics={
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "processing_time": 1.23
    }
)
```

##### collect_metrics(component_name: str, metric_name: str, value: float, metric_type: MetricType = MetricType.GAUGE, labels: Optional[Dict] = None)
收集组件指标

**参数**:
- `component_name` (str): 组件名称
- `metric_name` (str): 指标名称
- `value` (float): 指标值
- `metric_type` (MetricType): 指标类型，默认为GAUGE
- `labels` (dict, optional): 指标标签

**返回**:
- `None`

**示例**:
```python
from src.features.monitoring import MetricType

# 收集性能指标
monitor.collect_metrics(
    component_name="feature_engineer",
    metric_name="processing_time",
    value=1.23,
    metric_type=MetricType.HISTOGRAM,
    labels={"operation": "technical_features"}
)

# 收集错误率指标
monitor.collect_metrics(
    component_name="feature_engineer",
    metric_name="error_rate",
    value=0.05,
    metric_type=MetricType.GAUGE
)
```

##### get_component_metrics(component_name: str)
获取组件指标

**参数**:
- `component_name` (str): 组件名称

**返回**:
- `Dict[str, Any]`: 组件指标字典

**示例**:
```python
metrics = monitor.get_component_metrics("feature_engineer")
print(f"CPU使用率: {metrics.get('cpu_usage', 0)}%")
print(f"内存使用率: {metrics.get('memory_usage', 0)}%")
```

##### get_all_metrics()
获取所有组件指标

**返回**:
- `Dict[str, Dict[str, Any]]`: 所有组件指标字典

**示例**:
```python
all_metrics = monitor.get_all_metrics()
for component, metrics in all_metrics.items():
    print(f"{component}: {metrics}")
```

##### get_component_status(component_name: str)
获取组件状态

**参数**:
- `component_name` (str): 组件名称

**返回**:
- `Optional[Dict[str, Any]]`: 组件状态信息

**示例**:
```python
status = monitor.get_component_status("feature_engineer")
if status:
    print(f"状态: {status['status']}")
    print(f"启动时间: {status['start_time']}")
```

##### get_all_status()
获取所有组件状态

**返回**:
- `Dict[str, Dict[str, Any]]`: 所有组件状态字典

**示例**:
```python
all_status = monitor.get_all_status()
for component, status in all_status.items():
    print(f"{component}: {status['status']}")
```

##### start_monitoring()
启动监控

**返回**:
- `None`

**示例**:
```python
monitor.start_monitoring()
print("监控已启动")
```

##### stop_monitoring()
停止监控

**返回**:
- `None`

**示例**:
```python
monitor.stop_monitoring()
print("监控已停止")
```

##### export_metrics(file_path: str)
导出指标数据

**参数**:
- `file_path` (str): 导出文件路径

**返回**:
- `None`

**示例**:
```python
monitor.export_metrics("metrics_export.json")
```

##### get_performance_report()
获取性能报告

**返回**:
- `Dict[str, Any]`: 性能报告字典

**示例**:
```python
report = monitor.get_performance_report()
print(f"总组件数: {report['total_components']}")
print(f"运行中组件: {report['running_components']}")
print(f"平均CPU使用率: {report['avg_cpu_usage']}%")
```

## 配置选项

### 基础监控配置
```python
basic_config = {
    "monitor_interval": 5.0,        # 监控间隔（秒）
    "max_history_size": 1000,       # 最大历史记录数
    "enable_system_metrics": True,   # 启用系统指标
    "enable_performance_analysis": True  # 启用性能分析
}
```

### 阈值配置
```python
thresholds_config = {
    "cpu_usage": 80.0,      # CPU使用率阈值
    "memory_usage": 80.0,   # 内存使用率阈值
    "error_rate": 5.0,      # 错误率阈值
    "response_time": 5.0,   # 响应时间阈值
    "throughput": 1000      # 吞吐量阈值
}
```

### 告警配置
```python
alert_config = {
    "enable_alerts": True,           # 启用告警
    "alert_channels": ["console", "email"],  # 告警通道
    "suppression_time": 300,         # 告警抑制时间（秒）
    "escalation_enabled": True       # 启用告警升级
}
```

## 使用示例

### 基础使用
```python
from src.features.monitoring import FeaturesMonitor, MetricType

# 创建监控器
monitor = FeaturesMonitor()

# 注册组件
monitor.register_component("feature_engineer", "processor")
monitor.register_component("technical_processor", "processor")

# 启动监控
monitor.start_monitoring()

# 收集指标
monitor.collect_metrics(
    "feature_engineer",
    "processing_time",
    1.23,
    MetricType.HISTOGRAM
)

# 获取报告
report = monitor.get_performance_report()
print(report)
```

### 高级使用
```python
# 自定义配置
config = {
    "monitor_interval": 10.0,
    "thresholds": {
        "cpu_usage": 70.0,
        "memory_usage": 75.0,
        "error_rate": 3.0
    },
    "alert_config": {
        "enable_alerts": True,
        "alert_channels": ["console", "email", "webhook"]
    }
}

monitor = FeaturesMonitor(config=config)

# 注册多个组件
components = [
    ("feature_engineer", "processor"),
    ("technical_processor", "processor"),
    ("feature_selector", "processor"),
    ("feature_standardizer", "processor")
]

for name, component_type in components:
    monitor.register_component(name, component_type)

# 模拟指标收集
import time
import random

monitor.start_monitoring()

for i in range(10):
    # 模拟CPU使用率
    cpu_usage = random.uniform(20, 90)
    monitor.collect_metrics("feature_engineer", "cpu_usage", cpu_usage)
    
    # 模拟内存使用率
    memory_usage = random.uniform(30, 85)
    monitor.collect_metrics("feature_engineer", "memory_usage", memory_usage)
    
    # 模拟处理时间
    processing_time = random.uniform(0.5, 3.0)
    monitor.collect_metrics("feature_engineer", "processing_time", processing_time, MetricType.HISTOGRAM)
    
    time.sleep(1)

# 获取详细报告
report = monitor.get_performance_report()
print("性能报告:")
print(f"总组件数: {report['total_components']}")
print(f"运行中组件: {report['running_components']}")
print(f"平均CPU使用率: {report['avg_cpu_usage']:.2f}%")
print(f"平均内存使用率: {report['avg_memory_usage']:.2f}%")
print(f"平均处理时间: {report['avg_processing_time']:.3f}秒")

# 导出指标
monitor.export_metrics("feature_monitoring_metrics.json")
```

### 上下文管理器使用
```python
# 使用上下文管理器自动启动和停止监控
with FeaturesMonitor() as monitor:
    monitor.register_component("feature_engineer", "processor")
    
    # 执行特征工程任务
    # ... 特征工程代码 ...
    
    # 收集指标
    monitor.collect_metrics("feature_engineer", "task_completed", 1, MetricType.COUNTER)
    
    # 获取最终报告
    report = monitor.get_performance_report()
    print("任务完成报告:", report)
```

## 装饰器使用

### monitor_operation装饰器
```python
from src.features.monitoring import monitor_operation

@monitor_operation("feature_engineer", "generate_features")
def generate_technical_features(stock_data, indicators=None):
    # 特征生成逻辑
    # ...
    return features

# 使用装饰的函数会自动收集性能指标
features = generate_technical_features(data, ["ma", "rsi"])
```

## 性能优化建议

### 1. 监控间隔设置
- 根据系统负载调整监控间隔
- 高负载时使用较长间隔（10-30秒）
- 低负载时使用较短间隔（1-5秒）

### 2. 指标收集优化
- 使用批量收集减少I/O开销
- 合理设置指标历史记录大小
- 定期清理过期指标数据

### 3. 告警配置
- 设置合理的阈值避免误报
- 配置告警抑制时间
- 使用多通道告警确保及时通知

### 4. 内存管理
- 定期清理历史指标数据
- 使用数据压缩减少内存占用
- 设置合理的内存限制

## 故障排除

### 常见问题

#### 1. 监控启动失败
**问题**: `RuntimeError: 监控线程启动失败`
**解决方案**: 检查系统资源，确保有足够的线程和内存

#### 2. 指标收集异常
**问题**: `ValueError: 无效的指标值`
**解决方案**: 确保指标值为数值类型，检查指标名称格式

#### 3. 告警不触发
**问题**: 告警阈值设置后未触发
**解决方案**: 检查阈值设置，确保指标收集正常

#### 4. 性能影响
**问题**: 监控对系统性能影响较大
**解决方案**: 调整监控间隔，优化指标收集逻辑

### 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 检查组件状态
```python
# 检查所有组件状态
all_status = monitor.get_all_status()
for component, status in all_status.items():
    print(f"{component}: {status}")
```

#### 3. 验证指标收集
```python
# 检查特定组件指标
metrics = monitor.get_component_metrics("feature_engineer")
print("组件指标:", metrics)
```

## 版本历史

### v1.0.0
- 初始版本
- 基础监控功能
- 组件注册和管理
- 指标收集

### v1.1.0
- 添加告警功能
- 增强性能分析
- 支持多种指标类型
- 添加导出功能

### v1.2.0
- 添加装饰器支持
- 增强配置管理
- 优化内存使用
- 完善文档

---

**文档版本**: 1.2.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 