# 监控告警系统使用指南

## 概述

监控告警系统是RQA2025项目基础设施层的核心组件，提供实时测试执行监控、性能指标监控和异常告警功能。系统采用模块化设计，支持多种告警规则、通知通道和监控策略。

## 核心功能

### 1. 性能监控
- **实时监控**: CPU、内存、磁盘、网络等系统资源使用情况
- **历史数据**: 保存历史性能指标，支持趋势分析
- **自定义指标**: 支持添加自定义性能指标

### 2. 测试执行监控
- **测试生命周期**: 监控测试的启动、执行、完成状态
- **超时检测**: 自动检测测试执行超时
- **执行统计**: 收集测试执行时间和结果统计

### 3. 智能告警
- **多级告警**: INFO、WARNING、ERROR、CRITICAL四个级别
- **规则引擎**: 支持自定义告警规则和阈值
- **冷却机制**: 防止告警频繁触发
- **告警分类**: 系统错误、资源耗尽、测试失败等

### 4. 通知管理
- **多渠道通知**: 邮件、Webhook、控制台、文件等
- **灵活配置**: 支持不同通知通道的个性化配置
- **异步处理**: 告警通知异步发送，不阻塞主流程

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  性能监控器     │    │  测试监控器     │    │  告警管理器     │
│PerformanceMonitor│    │TestExecutionMonitor│  │  AlertManager   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  通知管理器     │
                    │NotificationManager│
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  监控告警系统   │
                    │MonitoringAlertSystem│
                    └─────────────────┘
```

## 快速开始

### 1. 基本使用

```python
from src.infrastructure.performance import (
    create_monitoring_system, 
    start_monitoring,
    get_system_status
)

# 创建监控系统
system = create_monitoring_system()

# 启动监控
system.start()

# 注册测试
test_id = system.register_test("test_001", "性能测试")

# 更新测试状态
system.update_test_status(test_id, "completed", execution_time=30.5)

# 获取系统状态
status = system.get_system_status()
print(f"系统运行状态: {status['running']}")
print(f"活跃测试数量: {status['active_tests']}")
print(f"活跃告警数量: {status['active_alerts']}")

# 停止监控
system.stop()
```

### 2. 便捷函数

```python
from src.infrastructure.performance import (
    start_monitoring,
    get_system_status,
    get_performance_report
)

# 快速启动监控
system = start_monitoring()

# 获取系统状态
status = get_system_status(system)

# 获取性能报告
report = get_performance_report(system, minutes=60)
print(f"CPU使用率: {report['cpu_usage']['current']}%")
print(f"内存使用率: {report['memory_usage']['current']}%")

# 停止监控
system.stop()
```

## 详细功能说明

### 1. 性能监控器 (PerformanceMonitor)

#### 基本操作

```python
from src.infrastructure.performance import PerformanceMonitor

# 创建监控器
monitor = PerformanceMonitor(update_interval=5)  # 5秒更新一次

# 启动监控
monitor.start_monitoring()

# 获取当前指标
metrics = monitor.get_current_metrics()
print(f"CPU使用率: {metrics.cpu_usage}%")
print(f"内存使用率: {metrics.memory_usage}%")
print(f"磁盘使用率: {metrics.disk_usage}%")
print(f"网络延迟: {metrics.network_latency}ms")

# 获取历史数据
history = monitor.get_metrics_history(minutes=30)

# 停止监控
monitor.stop_monitoring()
```

#### 性能指标说明

- **cpu_usage**: CPU使用率百分比 (0-100)
- **memory_usage**: 内存使用率百分比 (0-100)
- **disk_usage**: 磁盘使用率百分比 (0-100)
- **network_latency**: 网络延迟毫秒数
- **test_execution_time**: 测试执行时间秒数
- **test_success_rate**: 测试成功率百分比 (0-100)
- **active_threads**: 活跃线程数量
- **timestamp**: 指标采集时间戳

### 2. 告警管理器 (AlertManager)

#### 告警规则配置

```python
from src.infrastructure.performance import (
    AlertManager, 
    AlertRule, 
    AlertType, 
    AlertLevel
)

alert_manager = AlertManager()

# 添加CPU使用率告警规则
cpu_rule = AlertRule(
    name="CPU使用率过高",
    alert_type=AlertType.SYSTEM_ERROR,
    alert_level=AlertLevel.WARNING,
    condition="cpu_usage > threshold",
    threshold=80.0,
    cooldown=300  # 5分钟冷却时间
)

alert_manager.add_alert_rule(cpu_rule)

# 添加内存使用率告警规则
memory_rule = AlertRule(
    name="内存使用率过高",
    alert_type=AlertType.RESOURCE_EXHAUSTION,
    alert_level=AlertLevel.WARNING,
    condition="memory_usage > threshold",
    threshold=85.0,
    cooldown=300
)

alert_manager.add_alert_rule(memory_rule)

# 添加测试超时告警规则
timeout_rule = AlertRule(
    name="测试执行超时",
    alert_type=AlertType.TEST_TIMEOUT,
    alert_level=AlertLevel.ERROR,
    condition="test_execution_time > threshold",
    threshold=300.0,  # 5分钟
    cooldown=60
)

alert_manager.add_alert_rule(timeout_rule)
```

#### 告警处理器注册

```python
def cpu_alert_handler(alert):
    """CPU告警处理器"""
    print(f"CPU告警: {alert.message}")
    print(f"当前值: {alert.details['current_value']}%")
    print(f"阈值: {alert.details['threshold']}%")

def memory_alert_handler(alert):
    """内存告警处理器"""
    print(f"内存告警: {alert.message}")
    # 可以在这里添加内存清理逻辑

# 注册告警处理器
alert_manager.register_alert_handler(AlertType.SYSTEM_ERROR, cpu_alert_handler)
alert_manager.register_alert_handler(AlertType.RESOURCE_EXHAUSTION, memory_alert_handler)
```

#### 告警管理操作

```python
# 检查告警条件
alert_manager.check_alerts(metrics)

# 获取活跃告警
active_alerts = alert_manager.get_active_alerts()
for alert in active_alerts:
    print(f"告警ID: {alert.id}")
    print(f"级别: {alert.alert_level.value}")
    print(f"类型: {alert.alert_type.value}")
    print(f"消息: {alert.message}")
    print(f"时间: {alert.timestamp}")

# 解决告警
alert_manager.resolve_alert("alert_id_001")

# 获取告警历史
history = alert_manager.get_alert_history(hours=24)
```

### 3. 通知管理器 (NotificationManager)

#### 通知通道配置

```python
from src.infrastructure.performance import NotificationManager

notification_manager = NotificationManager()

# 配置邮件通知
email_config = {
    "from_email": "monitor@example.com",
    "to_email": "admin@example.com",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": True,
    "username": "your_email@gmail.com",
    "password": "your_password"
}

notification_manager.register_channel(
    "email",
    notification_manager.send_email_notification,
    email_config
)

# 配置Webhook通知
webhook_config = {
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    "timeout": 10,
    "retry_count": 3
}

notification_manager.register_channel(
    "webhook",
    notification_manager.send_webhook_notification,
    webhook_config
)

# 配置控制台通知
notification_manager.register_channel(
    "console",
    lambda alert, config: print(f"控制台告警: {alert.message}")
)
```

#### 发送通知

```python
# 发送到指定通道
notification_manager.send_notification(alert, ["email", "webhook"])

# 发送到所有通道
notification_manager.send_notification(alert)
```

### 4. 测试执行监控器 (TestExecutionMonitor)

#### 测试生命周期管理

```python
from src.infrastructure.performance import TestExecutionMonitor

test_monitor = TestExecutionMonitor()

# 启动监控
test_monitor.start_monitoring()

# 注册测试
test_id = test_monitor.register_test("test_001", "性能基准测试")
print(f"测试已注册: {test_id}")

# 更新测试状态
test_monitor.update_test_status(
    test_id, 
    "completed", 
    execution_time=45.2,
    error_message=None
)

# 获取活跃测试
active_tests = test_monitor.get_active_tests()
for test in active_tests:
    print(f"测试ID: {test.test_id}")
    print(f"测试名称: {test.test_name}")
    print(f"状态: {test.status}")
    print(f"开始时间: {test.start_time}")

# 获取测试历史
history = test_monitor.get_test_history(hours=24)

# 停止监控
test_monitor.stop_monitoring()
```

## 配置说明

### 1. 配置文件结构

系统使用YAML配置文件进行配置，主要配置项包括：

```yaml
# 性能监控配置
performance_monitoring:
  update_interval: 5  # 指标更新间隔(秒)
  metrics_history_size: 1000  # 历史数据保存数量
  network_latency_check_url: "http://www.baidu.com"

# 告警规则配置
alert_rules:
  cpu_usage:
    name: "CPU使用率过高"
    alert_type: "system_error"
    alert_level: "warning"
    condition: "cpu_usage > threshold"
    threshold: 80.0
    enabled: true
    cooldown: 300

# 通知配置
notifications:
  email:
    enabled: true
    from_email: "monitor@example.com"
    to_email: "admin@example.com"
    smtp_server: "localhost"
    use_tls: false

# 测试监控配置
test_monitoring:
  default_timeout: 300  # 默认超时时间(秒)
  timeout_check_interval: 1  # 超时检查间隔(秒)
```

### 2. 环境变量配置

系统支持通过环境变量进行配置：

```bash
# 性能监控配置
export MONITORING_UPDATE_INTERVAL=10
export MONITORING_METRICS_HISTORY_SIZE=2000

# 告警配置
export ALERT_CPU_THRESHOLD=75.0
export ALERT_MEMORY_THRESHOLD=80.0

# 通知配置
export NOTIFICATION_EMAIL_ENABLED=true
export NOTIFICATION_EMAIL_TO=admin@company.com
export NOTIFICATION_WEBHOOK_URL=https://hooks.slack.com/...
```

## 最佳实践

### 1. 告警规则设计

- **合理设置阈值**: 避免过于敏感或迟钝的告警
- **分级告警**: 根据严重程度设置不同级别
- **冷却时间**: 防止告警频繁触发
- **告警聚合**: 相似告警进行聚合处理

### 2. 性能监控优化

- **监控频率**: 根据系统负载调整监控频率
- **数据保留**: 合理设置历史数据保留时间
- **资源消耗**: 监控系统本身的资源消耗
- **异常处理**: 完善监控异常的处理机制

### 3. 通知管理

- **多渠道备份**: 重要告警使用多个通知通道
- **通知内容**: 告警信息要包含足够的上下文
- **响应时间**: 设置合理的通知响应时间
- **通知过滤**: 避免无关告警的干扰

### 4. 系统集成

- **CI/CD集成**: 在CI/CD流程中集成监控告警
- **日志管理**: 与现有日志系统集成
- **仪表板**: 提供Web监控仪表板
- **API接口**: 提供REST API接口

## 故障排除

### 1. 常见问题

#### 告警不触发
- 检查告警规则是否启用
- 验证阈值设置是否合理
- 确认监控数据是否正常收集
- 检查冷却时间是否过长

#### 通知发送失败
- 验证通知通道配置
- 检查网络连接和防火墙设置
- 确认认证信息是否正确
- 查看系统日志中的错误信息

#### 性能监控异常
- 检查系统资源是否充足
- 验证依赖库是否正确安装
- 确认监控权限是否足够
- 查看监控线程是否正常运行

### 2. 日志分析

系统提供详细的日志记录，可以通过日志分析问题：

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 查看监控系统日志
monitoring_logger = logging.getLogger('monitoring_system')
```

### 3. 性能调优

- **监控频率**: 根据系统负载调整监控频率
- **数据存储**: 优化历史数据的存储策略
- **告警处理**: 优化告警处理的并发性能
- **资源管理**: 合理管理监控系统的资源消耗

## 扩展开发

### 1. 自定义告警规则

```python
class CustomAlertRule(AlertRule):
    def __init__(self, name, threshold, custom_condition):
        super().__init__(
            name=name,
            alert_type=AlertType.CUSTOM,
            alert_level=AlertLevel.WARNING,
            condition=custom_condition,
            threshold=threshold
        )
    
    def evaluate(self, metrics, test_info):
        # 实现自定义评估逻辑
        return self._custom_evaluation(metrics, test_info)

# 使用自定义规则
custom_rule = CustomAlertRule("自定义告警", 90.0, "custom_condition")
alert_manager.add_alert_rule(custom_rule)
```

### 2. 自定义通知通道

```python
def custom_notification_handler(alert, config):
    """自定义通知处理器"""
    # 实现自定义通知逻辑
    print(f"自定义通知: {alert.message}")
    
    # 可以发送到自定义系统
    send_to_custom_system(alert, config)

# 注册自定义通知通道
notification_manager.register_channel(
    "custom",
    custom_notification_handler,
    {"custom_config": "value"}
)
```

### 3. 自定义性能指标

```python
class CustomPerformanceMetrics(PerformanceMetrics):
    def __init__(self):
        super().__init__()
        self.custom_metric = 0.0
    
    def collect_custom_metric(self):
        """收集自定义指标"""
        # 实现自定义指标收集逻辑
        self.custom_metric = calculate_custom_metric()
```

## 总结

监控告警系统为RQA2025项目提供了完整的测试执行监控和异常告警解决方案。通过合理的配置和使用，可以有效提升测试质量和系统稳定性。系统采用模块化设计，支持灵活扩展和定制，满足不同场景的监控需求。

## 相关链接

- [性能测试框架指南](./PERFORMANCE_TEST_OPTIMIZATION_GUIDE.md)
- [自动化测试流水线指南](./AUTOMATED_TEST_PIPELINE_GUIDE.md)
- [CI/CD集成指南](./CICD_INTEGRATION_GUIDE.md)
- [分布式测试执行指南](./DISTRIBUTED_TEST_EXECUTION_GUIDE.md)
- [测试报告增强指南](./TEST_REPORT_ENHANCEMENT_GUIDE.md)
