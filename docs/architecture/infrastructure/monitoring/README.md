# 监控与可观测性（monitoring/observability）架构设计说明

## 1. 模块定位
monitoring模块为RQA2025系统所有主流程、子系统提供系统级、应用级、业务级、性能级、异常级的全方位监控与告警能力，是保障系统可用性、可追溯性、可观测性的核心。

## 2. 主要子系统
- **系统级监控**：SystemMonitor 采集CPU、内存、磁盘、网络等系统资源，支持Prometheus集成与告警。
- **应用级监控**：ApplicationMonitor 监控函数调用、错误、慢执行、健康检查，支持Prometheus、InfluxDB等多后端。
- **性能监控**：PerformanceMonitor 采集各模块性能指标，支持多种存储后端与Prometheus集成。
- **行为与业务监控**：BehaviorMonitor 实时检测交易行为异常（如乌龙指、幌骗等），支持风控联动。
- **模型与回测监控**：ModelMonitor、BacktestMonitor 监控模型性能、漂移、回测绩效等。
- **存储与灾备监控**：StorageMonitor、DisasterMonitor 监控存储、灾备状态与同步。
- **统一指标收集与可视化**：MetricsCollector、PrometheusMonitor、MonitoringSystem 支持指标注册、采集、告警、Prometheus/PushGateway集成、可视化。

## 3. 典型用法
### 系统与应用监控
```python
from src.infrastructure.monitoring import SystemMonitor, ApplicationMonitor
sysmon = SystemMonitor()
appmon = ApplicationMonitor(app_name='rqa2025')
```

### 性能与行为监控
```python
from src.infrastructure.monitoring import PerformanceMonitor, BehaviorMonitor
perfmon = PerformanceMonitor(config)
behavmon = BehaviorMonitor()
```

### 模型与回测监控
```python
from src.infrastructure.monitoring import ModelMonitor, BacktestMonitor
modelmon = ModelMonitor()
btmon = BacktestMonitor()
```

### 指标收集与Prometheus集成
```python
from src.infrastructure.monitoring import MetricsCollector, PrometheusMonitor
metrics = MetricsCollector()
prom = PrometheusMonitor()
```

## 4. 在主流程中的地位
- 为所有主流程、子系统提供系统级、应用级、业务级、性能级的全方位监控与告警能力，保障系统可用性、可追溯性、可观测性。
- 支持Prometheus、InfluxDB、PushGateway等多种监控后端与可视化平台，便于运维与业务联动。
- 接口抽象与注册机制，便于扩展新监控项、适配新后端、Mock测试等。

## 5. 测试与质量保障
- 已实现高质量pytest单元测试，覆盖系统监控、应用监控、性能监控、行为监控、模型/回测监控、存储/灾备监控、统一指标收集等主要功能和边界。
- 测试用例见：tests/unit/infrastructure/monitoring/ 目录下相关文件。 