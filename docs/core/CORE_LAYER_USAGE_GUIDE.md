# 核心层使用指南

## 版本信息
- **版本**: 3.0.0
- **日期**: 2025-08-08
- **状态**: 正式发布

## 1. 概述

核心层是RQA2025系统的基础架构层，提供了统一的基础组件、依赖注入容器、事件总线、业务流程编排器等核心功能。本指南将详细介绍如何使用这些组件。

## 2. 快速开始

### 2.1 基本导入

```python
from src.core import (
    # 基础组件
    BaseComponent,
    BaseService,
    ComponentStatus,
    ComponentHealth,
    ComponentInfo,
    generate_id,
    validate_config,
    retry_on_failure,
    
    # 核心服务
    EventBus,
    DependencyContainer,
    ServiceContainer,
    BusinessProcessOrchestrator,
    
    # 异常处理
    CoreException,
    EventBusException,
    ContainerException,
    OrchestratorException,
    ServiceException,
    ConfigurationException,
    ValidationException,
    StateTransitionException,
    HealthCheckException,
    
    # 架构层
    CoreServicesLayer,
    InfrastructureLayer,
    DataManagementLayer,
    FeatureProcessingLayer,
    ModelInferenceLayer,
    StrategyDecisionLayer,
    RiskComplianceLayer,
    TradingExecutionLayer,
    MonitoringFeedbackLayer,
    
    # 层接口
    LayerInterface,
    BaseLayerImplementation,
    InterfaceFactory,
    DataManagementInterface,
    FeatureProcessingInterface,
    ModelInferenceInterface,
    StrategyDecisionInterface,
    RiskComplianceInterface,
    TradingExecutionInterface,
    MonitoringFeedbackInterface,
    InfrastructureInterface,
    CoreServicesInterface
)
```

### 2.2 基本使用示例

```python
# 创建依赖注入容器
container = DependencyContainer(enable_health_monitoring=True)
container.initialize()

# 注册服务
container.register_singleton("my_service", MyService)

# 解析服务
service = container.resolve("my_service")

# 创建事件总线
event_bus = EventBus(max_workers=10, enable_async=True)
event_bus.initialize()

# 订阅事件
def event_handler(event):
    print(f"收到事件: {event.event_type}")

event_bus.subscribe("data_collected", event_handler)

# 发布事件
event_bus.publish("data_collected", {"data": "test"})

# 创建业务流程编排器
orchestrator = BusinessProcessOrchestrator()
orchestrator.initialize()

# 关闭组件
event_bus.shutdown()
container.shutdown()
orchestrator.shutdown()
```

## 3. 基础组件

### 3.1 BaseComponent

`BaseComponent` 是所有核心组件的基础抽象类，提供了标准的生命周期管理。

```python
from src.core import BaseComponent, ComponentStatus, ComponentHealth

class MyComponent(BaseComponent):
    def __init__(self):
        super().__init__("MyComponent", "1.0.0", "我的组件")
    
    def _initialize_impl(self) -> bool:
        """实现初始化逻辑"""
        # 初始化代码
        return True
    
    def _start_impl(self) -> bool:
        """实现启动逻辑"""
        # 启动代码
        return True
    
    def _stop_impl(self) -> bool:
        """实现停止逻辑"""
        # 停止代码
        return True
    
    def shutdown(self) -> bool:
        """实现关闭逻辑"""
        # 关闭代码
        return True
    
    def health_check(self) -> bool:
        """健康检查"""
        return self.get_health() == ComponentHealth.HEALTHY

# 使用组件
component = MyComponent()
component.initialize()
component.start()

# 检查状态
status = component.get_status()
health = component.get_health()
info = component.get_info()
```

### 3.2 BaseService

`BaseService` 是服务组件的基类，支持依赖注入和配置管理。

```python
from src.core import BaseService

class MyService(BaseService):
    def __init__(self):
        super().__init__("MyService", "1.0.0", "我的服务")
        self.add_dependency("database")
        self.set_config("timeout", 30)
    
    def _initialize_impl(self) -> bool:
        # 初始化服务
        return True
    
    def shutdown(self) -> bool:
        # 关闭服务
        return True

# 使用服务
service = MyService()
service.initialize()
```

## 4. 依赖注入容器

### 4.1 基本概念

依赖注入容器支持三种生命周期：
- **SINGLETON**: 单例模式，整个应用生命周期内只有一个实例
- **TRANSIENT**: 瞬时模式，每次解析都创建新实例
- **SCOPED**: 作用域模式，在同一个作用域内共享实例

### 4.2 服务注册

#### 4.2.1 注册单例服务

```python
# 注册类
container.register_singleton("database", DatabaseService)

# 注册实例
container.register_singleton("config", ConfigService())

# 注册工厂
def create_service():
    return MyService()

container.register_factory("my_service", create_service)
```

#### 4.2.2 注册瞬时服务

```python
container.register_transient("logger", LoggerService)
```

#### 4.2.3 注册作用域服务

```python
container.register_scoped("user_session", UserSessionService)
```

### 4.3 依赖注入

```python
from src.core import service, Lifecycle

@service("user_service", lifecycle=Lifecycle.SINGLETON)
class UserService:
    def __init__(self, database_service, logger_service):
        self.database = database_service
        self.logger = logger_service
    
    def get_user(self, user_id: str):
        return self.database.query(f"SELECT * FROM users WHERE id = {user_id}")

# 自动注册和依赖注入
user_service = container.resolve("user_service")
```

### 4.4 服务健康检查

```python
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

container.register_singleton("my_service", MyService, health_check=health_check)

# 检查服务健康状态
health = container.check_health("my_service")
```

## 5. 事件总线

### 5.1 基本使用

```python
# 创建事件总线
event_bus = EventBus(
    max_workers=10,
    enable_async=True,
    enable_persistence=True,
    enable_retry=True,
    enable_monitoring=True
)
event_bus.initialize()

# 订阅事件
def data_handler(event):
    print(f"处理数据事件: {event.data}")

event_bus.subscribe("data_collected", data_handler, priority=EventPriority.HIGH)

# 发布事件
event_id = event_bus.publish(
    "data_collected",
    {"symbol": "AAPL", "price": 150.0},
    source="data_collector",
    priority=EventPriority.NORMAL
)
```

### 5.2 异步事件处理

```python
import asyncio

async def async_handler(event):
    await asyncio.sleep(1)
    print(f"异步处理事件: {event.event_type}")

event_bus.subscribe_async("data_processed", async_handler)
```

### 5.3 事件历史查询

```python
# 获取事件历史
history = event_bus.get_event_history(
    event_type="data_collected",
    start_time=time.time() - 3600,  # 1小时前
    limit=100
)

# 获取事件统计
stats = event_bus.get_event_statistics()
```

## 6. 业务流程编排器

### 6.1 基本使用

```python
# 创建业务流程编排器
orchestrator = BusinessProcessOrchestrator(
    config_dir="config/processes",
    max_instances=100
)
orchestrator.initialize()

# 启动交易周期
process_id = orchestrator.start_trading_cycle(
    symbols=["AAPL", "GOOGL"],
    strategy_config={
        "strategy_type": "momentum",
        "parameters": {"lookback": 20}
    }
)

# 检查流程状态
status = orchestrator.get_current_state()
```

### 6.2 流程控制

```python
# 暂停流程
orchestrator.pause_process(process_id)

# 恢复流程
orchestrator.resume_process(process_id)

# 获取运行中的流程
running_processes = orchestrator.get_running_processes()
```

### 6.3 流程监控

```python
# 获取流程指标
metrics = orchestrator.get_process_metrics()

# 获取内存使用
memory_usage = orchestrator.get_memory_usage()

# 优化内存
orchestrator.optimize_memory()
```

## 7. 架构层

### 7.1 核心服务层

```python
# 创建核心服务层
core_services = CoreServicesLayer()

# 获取事件总线
event_bus = core_services.get_event_bus()

# 获取依赖注入容器
container = core_services.get_dependency_container()

# 注册服务
core_services.register_service("my_service", MyService())
```

### 7.2 基础设施层

```python
# 创建基础设施层
infrastructure = InfrastructureLayer(core_services)

# 获取配置
config = infrastructure.get_config("database_url")

# 设置缓存
infrastructure.set_cache("user_data", user_data, ttl=3600)

# 获取缓存
cached_data = infrastructure.get_cache("user_data")
```

### 7.3 数据管理层

```python
# 创建数据管理层
data_layer = DataManagementLayer(infrastructure)

# 收集市场数据
market_data = data_layer.collect_market_data(["AAPL", "GOOGL"])

# 检查数据质量
quality_result = data_layer.check_data_quality(market_data)

# 存储数据
data_layer.store_data(market_data)
```

## 8. 异常处理

### 8.1 异常类型

```python
from src.core.exceptions import (
    CoreException,
    EventBusException,
    ContainerException,
    OrchestratorException,
    ServiceException,
    ConfigurationException,
    ValidationException,
    StateTransitionException,
    HealthCheckException
)

# 处理异常
try:
    service = container.resolve("non_existent_service")
except ContainerException as e:
    print(f"容器异常: {e}")
except CoreException as e:
    print(f"核心异常: {e}")
```

### 8.2 自定义异常

```python
class MyCustomException(CoreException):
    def __init__(self, message: str, error_code: str = "CUSTOM_ERROR"):
        super().__init__(message, error_code)
```

## 9. 性能优化

### 9.1 内存优化

```python
# 使用内存优化器
from src.core.optimizations import MemoryOptimizer

optimizer = MemoryOptimizer()
optimizer.analyze_memory_usage()
optimizer.optimize_memory_allocation()
optimizer.optimize_garbage_collection()
```

### 9.2 性能监控

```python
# 使用性能监控器
from src.core.optimizations import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()

# 获取性能指标
metrics = monitor.get_performance_metrics()
```

## 10. 测试

### 10.1 单元测试

```python
import pytest
from src.core import EventBus, DependencyContainer

def test_event_bus():
    event_bus = EventBus()
    event_bus.initialize()
    
    events = []
    def handler(event):
        events.append(event)
    
    event_bus.subscribe("test_event", handler)
    event_bus.publish("test_event", {"data": "test"})
    
    assert len(events) == 1
    assert events[0].data["data"] == "test"
    
    event_bus.shutdown()

def test_dependency_container():
    container = DependencyContainer()
    container.initialize()
    
    container.register_singleton("test_service", TestService)
    service = container.resolve("test_service")
    
    assert isinstance(service, TestService)
    
    container.shutdown()
```

### 10.2 集成测试

```python
def test_core_integration():
    # 创建核心组件
    container = DependencyContainer()
    event_bus = EventBus()
    orchestrator = BusinessProcessOrchestrator()
    
    # 初始化
    container.initialize()
    event_bus.initialize()
    orchestrator.initialize()
    
    # 测试集成
    # ...
    
    # 清理
    container.shutdown()
    event_bus.shutdown()
    orchestrator.shutdown()
```

## 11. 最佳实践

### 11.1 组件设计

1. **继承BaseComponent**: 所有核心组件都应继承BaseComponent
2. **实现生命周期方法**: 正确实现initialize、start、stop、shutdown方法
3. **健康检查**: 实现health_check方法
4. **异常处理**: 使用统一的异常类型

### 11.2 事件驱动

1. **事件命名**: 使用清晰的事件类型命名
2. **事件优先级**: 合理设置事件优先级
3. **异步处理**: 对于耗时操作使用异步事件处理
4. **错误处理**: 实现事件重试机制

### 11.3 依赖注入

1. **生命周期选择**: 根据使用场景选择合适的生命周期
2. **循环依赖**: 避免循环依赖
3. **服务发现**: 使用自动服务发现功能
4. **健康检查**: 为关键服务实现健康检查

### 11.4 业务流程

1. **状态管理**: 使用状态机管理业务流程
2. **错误恢复**: 实现错误恢复和回滚机制
3. **监控**: 实时监控流程状态和性能
4. **资源管理**: 合理管理内存和资源使用

## 12. 故障排除

### 12.1 常见问题

1. **组件初始化失败**: 检查依赖服务和配置
2. **事件丢失**: 检查事件订阅和发布逻辑
3. **内存泄漏**: 使用内存优化器分析
4. **性能问题**: 使用性能监控器分析

### 12.2 调试技巧

1. **日志记录**: 启用详细日志记录
2. **健康检查**: 定期检查组件健康状态
3. **性能监控**: 监控关键性能指标
4. **事件追踪**: 使用事件历史追踪问题

---

**最后更新**: 2025-08-08  
**维护者**: RQA2025 Team  
**状态**: ✅ 活跃维护  
**版本**: 3.0.0
