# 基础设施层接口定义API文档

## 📊 文档信息

**文档版本**: v1.0  
**创建日期**: 2025-10-24  
**适用模块**: `src\infrastructure\interfaces`  
**文档类型**: API参考文档

---

## 🎯 概述

基础设施层接口定义提供了RQA2025系统所有基础设施服务的标准化接口规范，采用Python Protocol和ABC定义，确保类型安全和接口一致性。

### 核心特性

- ✅ **类型安全**: 完整的类型注解，支持IDE智能提示
- ✅ **接口隔离**: 遵循SOLID原则，接口职责单一
- ✅ **协议驱动**: 使用Protocol实现鸭子类型
- ✅ **标准化**: 统一的接口设计规范

---

## 📚 接口清单

### infrastructure_services.py - 基础设施服务接口

#### 1. IConfigManager - 配置管理接口

**功能**: 提供统一的配置管理服务

```python
class IConfigManager(Protocol):
    """配置管理接口
    
    提供配置的读取、写入、验证和监控功能。
    """
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔的层级访问 (如 "database.host")
            default: 默认值，键不存在时返回
            
        Returns:
            配置值或默认值
            
        Example:
            >>> config_manager.get("database.host", "localhost")
            'localhost'
        """
        ...
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            
        Returns:
            是否设置成功
            
        Example:
            >>> config_manager.set("database.port", 5432)
            True
        """
        ...
    
    def reload(self) -> bool:
        """
        重新加载配置
        
        Returns:
            是否重载成功
            
        Example:
            >>> config_manager.reload()
            True
        """
        ...
    
    def validate_config(self, config: Dict[str, Any] = None) -> bool:
        """
        验证配置
        
        Args:
            config: 待验证的配置字典，None表示验证当前配置
            
        Returns:
            配置是否有效
            
        Example:
            >>> config_manager.validate_config({"database": {"port": 5432}})
            True
        """
        ...
```

**使用示例**:

```python
from src.infrastructure.interfaces import IConfigManager
from src.infrastructure.core import InfrastructureServiceProvider

# 获取配置管理器
provider = InfrastructureServiceProvider()
config_manager: IConfigManager = provider.config_manager

# 读取配置
db_host = config_manager.get("database.host", "localhost")
db_port = config_manager.get("database.port", 5432)

# 设置配置
config_manager.set("cache.ttl", 3600)

# 验证配置
if config_manager.validate_config():
    print("配置有效")

# 重新加载配置
config_manager.reload()
```

---

#### 2. ICacheService - 缓存服务接口

**功能**: 提供统一的缓存服务

```python
class ICacheService(Protocol):
    """缓存服务接口
    
    提供缓存的增删改查和统计功能。
    """
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，不存在返回None
            
        Example:
            >>> cache.get("user:123")
            {'id': 123, 'name': 'John'}
        """
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示使用默认TTL
            
        Returns:
            是否设置成功
            
        Example:
            >>> cache.set("user:123", {'id': 123}, ttl=3600)
            True
        """
        ...
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
            
        Example:
            >>> cache.delete("user:123")
            True
        """
        ...
    
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            缓存是否存在
            
        Example:
            >>> cache.exists("user:123")
            True
        """
        ...
    
    def clear(self) -> bool:
        """
        清空所有缓存
        
        Returns:
            是否清空成功
            
        Example:
            >>> cache.clear()
            True
        """
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典，包含命中率、大小等
            
        Example:
            >>> cache.get_stats()
            {
                'hit_rate': 0.85,
                'size': 1024,
                'memory_usage': '10MB'
            }
        """
        ...
```

**使用示例**:

```python
from src.infrastructure.interfaces import ICacheService

# 获取缓存服务
cache: ICacheService = provider.cache_service

# 设置缓存
cache.set("user:123", {'id': 123, 'name': 'John'}, ttl=3600)

# 获取缓存
user = cache.get("user:123")

# 检查存在
if cache.exists("user:123"):
    print("缓存存在")

# 删除缓存
cache.delete("user:123")

# 查看统计
stats = cache.get_stats()
print(f"缓存命中率: {stats['hit_rate']}")

# 清空缓存
cache.clear()
```

---

#### 3. ILogger - 日志服务接口

**功能**: 提供统一的日志记录服务

```python
class ILogger(Protocol):
    """日志服务接口
    
    提供标准的日志记录功能。
    """
    
    def debug(self, message: str, **kwargs) -> None:
        """
        记录DEBUG级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外的上下文信息
            
        Example:
            >>> logger.debug("数据库查询", query="SELECT * FROM users", duration=0.5)
        """
        ...
    
    def info(self, message: str, **kwargs) -> None:
        """
        记录INFO级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外的上下文信息
            
        Example:
            >>> logger.info("用户登录成功", user_id=123, ip="192.168.1.1")
        """
        ...
    
    def warning(self, message: str, **kwargs) -> None:
        """
        记录WARNING级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外的上下文信息
            
        Example:
            >>> logger.warning("缓存命中率低", hit_rate=0.5)
        """
        ...
    
    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """
        记录ERROR级别日志
        
        Args:
            message: 日志消息
            exc_info: 异常信息
            **kwargs: 额外的上下文信息
            
        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.error("操作失败", exc_info=e, operation="risky_operation")
        """
        ...
    
    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """
        记录CRITICAL级别日志
        
        Args:
            message: 日志消息
            exc_info: 异常信息
            **kwargs: 额外的上下文信息
            
        Example:
            >>> logger.critical("数据库连接失败", exc_info=e, db="PostgreSQL")
        """
        ...
```

**使用示例**:

```python
from src.infrastructure.interfaces import ILogger

# 获取日志服务
logger: ILogger = provider.logger

# 不同级别的日志
logger.debug("调试信息", variable=value)
logger.info("用户登录", user_id=123)
logger.warning("性能下降", response_time=2.5)

try:
    risky_operation()
except Exception as e:
    logger.error("操作失败", exc_info=e)
```

---

#### 4. IMonitor - 监控服务接口

**功能**: 提供系统监控和指标记录服务

```python
class IMonitor(Protocol):
    """监控服务接口
    
    提供指标记录、监控和告警功能。
    """
    
    def record_metric(self, name: str, value: float, 
                     metric_type: str = "gauge",
                     tags: Optional[Dict[str, str]] = None,
                     timestamp: Optional[float] = None,
                     unit: Optional[str] = None) -> bool:
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            metric_type: 指标类型 (gauge/counter/histogram)
            tags: 标签字典
            timestamp: 时间戳，None表示当前时间
            unit: 单位（如 ms, MB, %）
            
        Returns:
            是否记录成功
            
        Example:
            >>> monitor.record_metric(
            ...     "api.response_time",
            ...     125.5,
            ...     metric_type="histogram",
            ...     tags={"endpoint": "/api/v1/users"},
            ...     unit="ms"
            ... )
            True
        """
        ...
    
    def increment_counter(self, name: str, value: int = 1,
                         tags: Optional[Dict[str, str]] = None) -> bool:
        """
        递增计数器
        
        Args:
            name: 计数器名称
            value: 递增值，默认为1
            tags: 标签字典
            
        Returns:
            是否递增成功
            
        Example:
            >>> monitor.increment_counter("api.requests", tags={"method": "GET"})
            True
        """
        ...
    
    def record_histogram(self, name: str, value: float,
                        tags: Optional[Dict[str, str]] = None) -> bool:
        """
        记录直方图数据
        
        Args:
            name: 直方图名称
            value: 数据值
            tags: 标签字典
            
        Returns:
            是否记录成功
            
        Example:
            >>> monitor.record_histogram("db.query_time", 45.2, 
            ...                          tags={"table": "users"})
            True
        """
        ...
```

**使用示例**:

```python
from src.infrastructure.interfaces import IMonitor
import time

# 获取监控服务
monitor: IMonitor = provider.monitor

# 记录性能指标
start = time.time()
# ... 执行操作 ...
duration = (time.time() - start) * 1000

monitor.record_metric(
    "api.response_time",
    duration,
    metric_type="histogram",
    tags={"endpoint": "/api/v1/users", "method": "GET"},
    unit="ms"
)

# 递增计数器
monitor.increment_counter("api.requests", tags={"status": "200"})

# 记录直方图
monitor.record_histogram("db.query_time", duration, tags={"operation": "SELECT"})
```

---

#### 5. IHealthChecker - 健康检查接口

**功能**: 提供服务健康检查功能

```python
class IHealthChecker(Protocol):
    """健康检查接口
    
    提供服务健康状态检查和监控功能。
    """
    
    def check_health(self, service_name: Optional[str] = None) -> HealthCheckResult:
        """
        执行健康检查
        
        Args:
            service_name: 服务名称，None表示检查所有服务
            
        Returns:
            健康检查结果
            
        Example:
            >>> result = health_checker.check_health("database")
            >>> if result.healthy:
            ...     print("服务健康")
        """
        ...
    
    def is_healthy(self, service_name: Optional[str] = None) -> bool:
        """
        快速检查健康状态
        
        Args:
            service_name: 服务名称，None表示检查整体状态
            
        Returns:
            是否健康
            
        Example:
            >>> if health_checker.is_healthy("cache"):
            ...     print("缓存服务正常")
        """
        ...
    
    def get_health_history(self, service_name: str,
                          limit: int = 100) -> List[HealthCheckResult]:
        """
        获取健康检查历史
        
        Args:
            service_name: 服务名称
            limit: 返回记录数量
            
        Returns:
            健康检查历史列表
            
        Example:
            >>> history = health_checker.get_health_history("database", limit=10)
            >>> for result in history:
            ...     print(f"{result.timestamp}: {result.status}")
        """
        ...
```

**使用示例**:

```python
from src.infrastructure.interfaces import IHealthChecker

# 获取健康检查服务
health_checker: IHealthChecker = provider.health_checker

# 检查所有服务
overall_result = health_checker.check_health()
print(f"整体状态: {overall_result.status}")

# 检查特定服务
db_result = health_checker.check_health("database")
if db_result.healthy:
    print("数据库健康")
else:
    print(f"数据库异常: {db_result.message}")

# 快速检查
if health_checker.is_healthy():
    print("系统正常运行")

# 查看历史
history = health_checker.get_health_history("cache", limit=10)
for result in history:
    print(f"{result.timestamp}: {result.status} ({result.response_time}ms)")
```

---

#### 6. IEventBus - 事件总线接口

**功能**: 提供异步事件发布订阅服务

```python
class IEventBus(Protocol):
    """事件总线接口
    
    提供事件的发布、订阅和管理功能。
    """
    
    def publish(self, event: Event) -> bool:
        """
        发布事件
        
        Args:
            event: 事件对象
            
        Returns:
            是否发布成功
            
        Example:
            >>> event = Event(
            ...     event_type="config.changed",
            ...     data={"key": "database.host", "value": "new-host"}
            ... )
            >>> event_bus.publish(event)
            True
        """
        ...
    
    def subscribe(self, event_type: str,
                 handler: Callable[[Event], None],
                 filter: Optional[Dict[str, Any]] = None,
                 priority: int = 0) -> str:
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            filter: 事件过滤条件
            priority: 优先级，数值越大优先级越高
            
        Returns:
            订阅ID
            
        Example:
            >>> def on_config_changed(event):
            ...     print(f"配置变更: {event.data}")
            >>> 
            >>> subscription_id = event_bus.subscribe(
            ...     "config.changed",
            ...     on_config_changed,
            ...     priority=10
            ... )
        """
        ...
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_id: 订阅ID
            
        Returns:
            是否取消成功
            
        Example:
            >>> event_bus.unsubscribe(subscription_id)
            True
        """
        ...
```

**使用示例**:

```python
from src.infrastructure.interfaces import IEventBus, Event

# 获取事件总线
event_bus: IEventBus = provider.event_bus

# 定义事件处理器
def on_config_changed(event: Event):
    print(f"配置变更: {event.data}")
    # 处理配置变更...

# 订阅事件
sub_id = event_bus.subscribe("config.changed", on_config_changed, priority=10)

# 发布事件
event = Event(
    event_type="config.changed",
    data={"key": "cache.ttl", "old_value": 3600, "new_value": 7200}
)
event_bus.publish(event)

# 取消订阅
event_bus.unsubscribe(sub_id)
```

---

### standard_interfaces.py - 标准接口定义

#### 7. IServiceProvider - 服务提供者接口

**功能**: 提供统一的服务访问入口

```python
class IServiceProvider(Protocol):
    """服务提供者接口
    
    提供统一的服务注册、解析和管理功能。
    """
    
    def register(self, service_name: str, service_instance: Any,
                singleton: bool = True) -> bool:
        """
        注册服务
        
        Args:
            service_name: 服务名称
            service_instance: 服务实例
            singleton: 是否单例模式
            
        Returns:
            是否注册成功
            
        Example:
            >>> provider.register("cache", cache_service, singleton=True)
            True
        """
        ...
    
    def resolve(self, service_name: str) -> Optional[Any]:
        """
        解析服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务实例
            
        Example:
            >>> cache = provider.resolve("cache")
        """
        ...
    
    def has_service(self, service_name: str) -> bool:
        """
        检查服务是否存在
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务是否已注册
            
        Example:
            >>> if provider.has_service("cache"):
            ...     print("缓存服务可用")
        """
        ...
```

**使用示例**:

```python
from src.infrastructure.interfaces import IServiceProvider

# 创建服务提供者
provider: IServiceProvider = InfrastructureServiceProvider()

# 注册服务
provider.register("cache", cache_service, singleton=True)
provider.register("logger", logger_service, singleton=True)

# 解析服务
cache = provider.resolve("cache")
logger = provider.resolve("logger")

# 检查服务
if provider.has_service("cache"):
    cache = provider.resolve("cache")
    cache.set("key", "value")
```

---

## 🔧 数据类型定义

### HealthCheckResult - 健康检查结果

```python
@dataclass
class HealthCheckResult:
    """健康检查结果"""
    
    service_name: str           # 服务名称
    healthy: bool               # 是否健康
    status: str                 # 状态描述
    response_time: float        # 响应时间（秒）
    message: str                # 详细消息
    timestamp: float            # 检查时间戳
    details: Dict[str, Any]     # 详细信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'service_name': self.service_name,
            'healthy': self.healthy,
            'status': self.status,
            'response_time': self.response_time,
            'message': self.message,
            'timestamp': self.timestamp,
            'details': self.details
        }
```

### Event - 事件对象

```python
@dataclass
class Event:
    """事件对象"""
    
    event_type: str             # 事件类型
    data: Dict[str, Any]        # 事件数据
    timestamp: float            # 事件时间戳
    source: str                 # 事件来源
    event_id: str               # 事件ID
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'source': self.source,
            'event_id': self.event_id
        }
```

---

## 📖 最佳实践

### 1. 依赖注入模式

**推荐方式**: 构造函数注入

```python
class TradingEngine:
    def __init__(self, infrastructure_provider: IInfrastructureServiceProvider):
        """通过构造函数注入基础设施依赖"""
        self.logger = infrastructure_provider.logger
        self.cache = infrastructure_provider.cache_service
        self.monitor = infrastructure_provider.monitor
        self.config = infrastructure_provider.config_manager
    
    def execute_trade(self, order):
        """执行交易"""
        # 使用日志服务
        self.logger.info("开始执行交易", order_id=order.id)
        
        # 使用缓存服务
        cached_data = self.cache.get(f"order:{order.id}")
        
        # 使用监控服务
        self.monitor.increment_counter("trades.executed")
```

### 2. 接口类型注解

**推荐方式**: 使用Protocol类型

```python
from typing import Protocol
from src.infrastructure.interfaces import ILogger, ICacheService

class MyService:
    def __init__(self, logger: ILogger, cache: ICacheService):
        """使用Protocol类型注解，提高类型安全"""
        self.logger: ILogger = logger
        self.cache: ICacheService = cache
```

### 3. 错误处理

**推荐方式**: 优雅降级

```python
def safe_operation():
    try:
        result = cache.get("key")
        if result is None:
            # 缓存未命中，从数据库加载
            result = load_from_database()
            cache.set("key", result)
        return result
    except Exception as e:
        logger.error("缓存操作失败", exc_info=e)
        # 降级处理：直接从数据库加载
        return load_from_database()
```

---

## 🎯 使用指南

### 快速开始

```python
from src.infrastructure.core import InfrastructureServiceProvider
from src.infrastructure.interfaces import (
    IConfigManager,
    ICacheService,
    ILogger,
    IMonitor,
    IHealthChecker,
    IEventBus
)

# 1. 创建服务提供者
provider = InfrastructureServiceProvider()

# 2. 获取各种服务
config: IConfigManager = provider.config_manager
cache: ICacheService = provider.cache_service
logger: ILogger = provider.logger
monitor: IMonitor = provider.monitor
health: IHealthChecker = provider.health_checker
events: IEventBus = provider.event_bus

# 3. 使用服务
config.set("app.name", "RQA2025")
cache.set("user:1", {"name": "John"}, ttl=3600)
logger.info("应用启动")
monitor.record_metric("app.memory", 128.5, unit="MB")

# 4. 健康检查
if health.is_healthy():
    print("系统正常")
```

---

## 📝 更新日志

### v1.0 (2025-10-24)
- 创建初始版本
- 补充9个核心接口的详细文档
- 添加使用示例和最佳实践
- 完善类型注解说明

---

**文档维护者**: RQA2025团队  
**最后更新**: 2025-10-24  

---

*本文档提供了基础设施层接口的完整API参考，帮助开发者快速理解和使用基础设施服务。*
