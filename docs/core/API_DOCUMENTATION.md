# 核心层API文档

## 版本信息
- **版本**: 3.0.0
- **日期**: 2025-08-08
- **状态**: 正式发布

## 1. 概述

本文档详细描述了RQA2025系统核心层的API接口，包括事件总线、依赖注入容器、服务容器、业务流程编排器等核心组件。

## 2. 事件总线 (EventBus)

### 2.1 类定义

```python
class EventBus(BaseComponent):
    """事件总线 - 优化版"""
    
    def __init__(self, max_workers: int = 10, enable_async: bool = True, 
                 enable_persistence: bool = True, enable_retry: bool = True, 
                 enable_monitoring: bool = True, batch_size: int = 10,
                 max_queue_size: int = 10000):
        """
        初始化事件总线
        
        Args:
            max_workers: 最大工作线程数
            enable_async: 是否启用异步处理
            enable_persistence: 是否启用持久化
            enable_retry: 是否启用重试机制
            enable_monitoring: 是否启用监控
            batch_size: 批处理大小
            max_queue_size: 最大队列大小
        """
```

### 2.2 主要方法

#### 2.2.1 订阅事件

```python
def subscribe(self, event_type: Union[EventType, str], handler: Callable, 
              priority: EventPriority = EventPriority.NORMAL, 
              async_handler: bool = False,
              retry_on_failure: bool = True,
              max_retries: int = 3):
    """
    订阅事件
    
    Args:
        event_type: 事件类型
        handler: 事件处理器
        priority: 优先级
        async_handler: 是否为异步处理器
        retry_on_failure: 是否在失败时重试
        max_retries: 最大重试次数
    
    Returns:
        bool: 订阅是否成功
    """
```

#### 2.2.2 发布事件

```python
def publish(self, event_type: Union[EventType, str], data: Dict[str, Any] = None, 
            source: str = "system", priority: EventPriority = EventPriority.NORMAL,
            event_id: Optional[str] = None, correlation_id: Optional[str] = None) -> str:
    """
    发布事件
    
    Args:
        event_type: 事件类型
        data: 事件数据
        source: 事件源
        priority: 优先级
        event_id: 事件ID
        correlation_id: 关联ID
    
    Returns:
        str: 事件ID
    """
```

#### 2.2.3 获取事件历史

```python
def get_event_history(self, event_type: Optional[Union[EventType, str]] = None, 
                     start_time: Optional[float] = None, 
                     end_time: Optional[float] = None,
                     limit: Optional[int] = None) -> List[Event]:
    """
    获取事件历史
    
    Args:
        event_type: 事件类型过滤
        start_time: 开始时间
        end_time: 结束时间
        limit: 限制数量
    
    Returns:
        List[Event]: 事件列表
    """
```

### 2.3 事件类型

```python
class EventType(Enum):
    """事件类型枚举"""
    # 数据层事件
    DATA_READY = "data_ready"
    DATA_COLLECTED = "data_collected"
    DATA_QUALITY_CHECKED = "data_quality_checked"
    
    # 特征层事件
    FEATURE_EXTRACTED = "feature_extracted"
    FEATURES_EXTRACTED = "features_extracted"
    
    # 模型层事件
    MODEL_PREDICTION = "model_prediction"
    MODEL_PREDICTED = "model_predicted"
    
    # 策略层事件
    SIGNAL_GENERATED = "signal_generated"
    STRATEGY_DECISION_READY = "strategy_decision_ready"
    
    # 风控层事件
    RISK_CHECKED = "risk_checked"
    COMPLIANCE_VERIFIED = "compliance_verified"
    
    # 交易层事件
    ORDER_CREATED = "order_created"
    EXECUTION_COMPLETED = "execution_completed"
    
    # 监控层事件
    PERFORMANCE_ALERT = "performance_alert"
    BUSINESS_ALERT = "business_alert"
```

## 3. 依赖注入容器 (DependencyContainer)

### 3.1 类定义

```python
class DependencyContainer:
    """依赖注入容器"""
    
    def __init__(self):
        """初始化容器"""
```

### 3.2 主要方法

#### 3.2.1 注册服务

```python
def register(self, name: str, service: Any = None, factory: Callable = None,
             lifecycle: Lifecycle = Lifecycle.SINGLETON,
             health_check: Optional[Callable] = None):
    """
    注册服务
    
    Args:
        name: 服务名称
        service: 服务实例
        factory: 工厂函数
        lifecycle: 生命周期
        health_check: 健康检查函数
    """
```

#### 3.2.2 获取服务

```python
def get(self, name: str) -> Any:
    """
    获取服务
    
    Args:
        name: 服务名称
    
    Returns:
        Any: 服务实例
    
    Raises:
        ContainerException: 服务不存在或创建失败
    """
```

#### 3.2.3 检查服务是否存在

```python
def has(self, name: str) -> bool:
    """
    检查服务是否存在
    
    Args:
        name: 服务名称
    
    Returns:
        bool: 是否存在
    """
```

### 3.3 生命周期枚举

```python
class Lifecycle(Enum):
    """服务生命周期枚举"""
    SINGLETON = "singleton"  # 单例
    TRANSIENT = "transient"  # 瞬时
    SCOPED = "scoped"        # 作用域
```

## 4. 服务容器 (ServiceContainer)

### 4.1 类定义

```python
class ServiceContainer:
    """服务容器管理 - 增强版"""
    
    def __init__(self, config_dir: str = "config/services"):
        """
        初始化服务容器
        
        Args:
            config_dir: 配置目录
        """
```

### 4.2 主要方法

#### 4.2.1 注册服务

```python
def register_service(self, config: ServiceConfig) -> bool:
    """
    注册服务
    
    Args:
        config: 服务配置
    
    Returns:
        bool: 注册是否成功
    """
```

#### 4.2.2 获取服务

```python
def get_service(self, name: str) -> Optional[Any]:
    """
    获取服务
    
    Args:
        name: 服务名称
    
    Returns:
        Optional[Any]: 服务实例
    """
```

#### 4.2.3 获取服务状态

```python
def get_service_status(self, name: str) -> Optional[ServiceStatus]:
    """
    获取服务状态
    
    Args:
        name: 服务名称
    
    Returns:
        Optional[ServiceStatus]: 服务状态
    """
```

### 4.3 服务配置

```python
@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    service_type: Optional[Type] = None
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    version: str = "1.0.0"
    dependencies: List[str] = None
    health_check: Optional[Callable] = None
    health_check_interval: int = 300
    max_instances: int = 1
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    weight: int = 1
    enabled: bool = True
    auto_start: bool = True
    config: Dict[str, Any] = None
```

## 5. 业务流程编排器 (BusinessProcessOrchestrator)

### 5.1 类定义

```python
class BusinessProcessOrchestrator(BaseComponent):
    """业务流程编排器"""
    
    def __init__(self, config_dir: str = "config/processes", max_instances: int = 100):
        """
        初始化编排器
        
        Args:
            config_dir: 配置目录
            max_instances: 最大实例数
        """
```

### 5.2 主要方法

#### 5.2.1 启动交易周期

```python
def start_trading_cycle(self, symbols: List[str], strategy_config: dict, process_id: str = None) -> str:
    """
    启动交易周期
    
    Args:
        symbols: 交易标的列表
        strategy_config: 策略配置
        process_id: 流程ID
    
    Returns:
        str: 流程ID
    """
```

#### 5.2.2 获取当前状态

```python
def get_current_state(self) -> BusinessProcessState:
    """
    获取当前状态
    
    Returns:
        BusinessProcessState: 当前状态
    """
```

#### 5.2.3 获取运行中的流程

```python
def get_running_processes(self) -> List[ProcessInstance]:
    """
    获取运行中的流程
    
    Returns:
        List[ProcessInstance]: 流程实例列表
    """
```

### 5.3 业务流程状态

```python
class BusinessProcessState(Enum):
    """业务流程状态 - 增强版"""
    IDLE = "idle"                           # 空闲状态
    DATA_COLLECTING = "data_collecting"     # 数据采集中
    DATA_QUALITY_CHECKING = "data_quality_checking"  # 数据质量检查中
    FEATURE_EXTRACTING = "feature_extracting"  # 特征提取中
    MODEL_PREDICTING = "model_predicting"   # 模型预测中
    STRATEGY_DECIDING = "strategy_deciding" # 策略决策中
    SIGNAL_GENERATING = "signal_generating" # 信号生成中
    RISK_CHECKING = "risk_checking"         # 风险检查中
    COMPLIANCE_VERIFYING = "compliance_verifying"  # 合规验证中
    ORDER_GENERATING = "order_generating"   # 订单生成中
    ORDER_EXECUTING = "order_executing"     # 订单执行中
    MONITORING_FEEDBACK = "monitoring_feedback"  # 监控反馈中
    COMPLETED = "completed"                 # 完成状态
    ERROR = "error"                         # 错误状态
    PAUSED = "paused"                       # 暂停状态
```

## 6. 基础组件 (BaseComponent)

### 6.1 类定义

```python
class BaseComponent(ABC):
    """基础组件抽象类"""
    
    def __init__(self, name: str = None, version: str = "1.0.0"):
        """
        初始化基础组件
        
        Args:
            name: 组件名称
            version: 版本号
        """
```

### 6.2 主要方法

#### 6.2.1 初始化

```python
def initialize(self) -> bool:
    """
    初始化组件
    
    Returns:
        bool: 初始化是否成功
    """
```

#### 6.2.2 启动

```python
def start(self) -> bool:
    """
    启动组件
    
    Returns:
        bool: 启动是否成功
    """
```

#### 6.2.3 停止

```python
def stop(self) -> bool:
    """
    停止组件
    
    Returns:
        bool: 停止是否成功
    """
```

#### 6.2.4 关闭

```python
def shutdown(self) -> bool:
    """
    关闭组件
    
    Returns:
        bool: 关闭是否成功
    """
```

#### 6.2.5 健康检查

```python
def health_check(self) -> bool:
    """
    健康检查
    
    Returns:
        bool: 健康状态
    """
```

## 7. 异常类

### 7.1 核心异常

```python
class CoreException(Exception):
    """核心异常基类"""
    pass

class EventBusException(CoreException):
    """事件总线异常"""
    pass

class ContainerException(CoreException):
    """容器异常"""
    pass

class OrchestratorException(CoreException):
    """编排器异常"""
    pass

class ServiceException(CoreException):
    """服务异常"""
    pass
```

## 8. 使用示例

### 8.1 事件总线使用示例

```python
from src.core import EventBus, EventType, EventPriority

# 创建事件总线
event_bus = EventBus()
event_bus.initialize()

# 定义事件处理器
def data_handler(event):
    print(f"收到数据事件: {event.data}")

# 订阅事件
event_bus.subscribe(EventType.DATA_READY, data_handler)

# 发布事件
event_id = event_bus.publish(
    EventType.DATA_READY,
    {"symbol": "000001.SZ", "price": 10.5},
    priority=EventPriority.HIGH
)

# 关闭事件总线
event_bus.shutdown()
```

### 8.2 依赖注入容器使用示例

```python
from src.core import DependencyContainer, Lifecycle

# 创建容器
container = DependencyContainer()

# 注册服务
class DataService:
    def get_data(self):
        return "data"

container.register("data_service", DataService(), lifecycle=Lifecycle.SINGLETON)

# 获取服务
data_service = container.get("data_service")
```

### 8.3 业务流程编排器使用示例

```python
from src.core import BusinessProcessOrchestrator

# 创建编排器
orchestrator = BusinessProcessOrchestrator()
orchestrator.initialize()

# 启动交易周期
process_id = orchestrator.start_trading_cycle(
    symbols=["000001.SZ", "000002.SZ"],
    strategy_config={"type": "momentum", "period": 20}
)

# 获取当前状态
current_state = orchestrator.get_current_state()
print(f"当前状态: {current_state}")

# 关闭编排器
orchestrator.shutdown()
```

## 9. 最佳实践

### 9.1 事件总线最佳实践

1. **合理使用优先级**: 根据事件重要性设置合适的优先级
2. **避免阻塞处理器**: 长时间运行的处理逻辑应该使用异步处理
3. **错误处理**: 在事件处理器中添加适当的错误处理
4. **资源管理**: 及时取消不需要的订阅

### 9.2 依赖注入最佳实践

1. **生命周期管理**: 根据服务特性选择合适的生命周期
2. **循环依赖**: 避免服务间的循环依赖
3. **健康检查**: 为关键服务添加健康检查
4. **配置管理**: 使用配置文件管理服务配置

### 9.3 业务流程编排最佳实践

1. **状态管理**: 合理设计业务流程状态
2. **错误处理**: 在流程中添加适当的错误处理和回滚机制
3. **监控**: 对业务流程进行监控和日志记录
4. **资源管理**: 合理管理流程实例资源

## 10. 版本历史

### v3.0.0 (2025-08-08)
- 完成核心层优化
- 实现长期优化任务
- 完善测试覆盖
- 更新API文档

### v2.0.0 (2025-07-15)
- 实现中期优化任务
- 添加分布式支持
- 实现多级缓存

### v1.0.0 (2025-06-01)
- 初始版本发布
- 实现基础功能
- 完成短期优化任务
